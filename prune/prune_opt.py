import gc
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import trange

from utils import timeit
from .d2prune_utils import D2SparseGPT, D2Wanda
from .pruner_zero import PrunerZero
from .sparsegpt import SparseGPT
from .wanda import Wanda


class D2Prune_OPT:
    '''
    D2Prune:
    1. using 1st-order activation derivatives and 2nd-order weights derivatives for pruning metric
    2. attention awareness: q/k/v weights hybrid update (D2SparseGPT) or no-update (D2Wanda)
    '''

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = args.device  # 'cpu' or 'cuda:0
        self.sparsity_ratio = args.sparsity_ratio
        self.nsamples = args.nsamples
        self.target_layer_names = args.target_layer_names  # []
        self.d2_sparsegpt = args.d2_sparsegpt
        self.d2_wanda = args.d2_wanda
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = self.args.logger

    def init_model(self):
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.decoder.layers

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-2):
        self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                # count += (W==0).sum().item()
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                # sub_count += (W == 0).sum().item()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count) / total_params

    @staticmethod
    def check_outlier_mean(mask, threshold):
        W = mask
        count = 0
        total_params = 0
        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio

    @torch.no_grad()
    def get_layer_dynamic_sparsity(self, subset, gpts_layers, wrapped_layers, dsm='owl', granularity='per-block'):
        """
        Sparsity compensation
        Compensate for over-pruning caused by uniform sparsity due to different layer sensitivities, and balance sparsity.
        :param dsm:dynamic sparsity method-->global static adjustments
        :return:subset each layer sparsity
        """
        if dsm == "owl":
            # self.layer_wmetric = []
            if granularity == 'per-block':
                self.layer_outlier_ratios = []
                self.block_sizes = []

                for name in subset:
                    # W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1, -1)))
                    if name in self.target_layer_names:
                        gpts = wrapped_layers
                    else:
                        gpts = gpts_layers
                    W_metric = (torch.abs(gpts[name].layer.weight.data) ** 2) * (
                            gpts[name].scaler_row.reshape((1, -1)) ** (1))
                    if self.args.d2_wanda:
                        # (lambda_1 ywx)^(1/2)
                        # W_metric += (self.r1 ** (1/2)) * (self.y_scaler_col.reshape((-1, 1)) ** (1 / 2)) * (torch.abs(self.layer.weight.data) ** (1 / 2)) * self.delta_x_scaler_row.reshape((1, -1)) ** (1/2)
                        # # (lambda_2 w^2x^2)^1/2=sqrt(lambda_2) wx
                        # W_metric += -(self.r2 ** (1/2)) * (torch.abs(self.layer.weight.data)) * (self.delta_x_scaler_row.reshape((1, -1)) ** (1))  # 768,128

                        # # new-->not scaling to sqrt: correct
                        ## ywx
                        W_metric += (gpts[name].r1) * (gpts[name].y_scaler_col.reshape((-1, 1)) ** (1)) * (
                            torch.abs(gpts[name].layer.weight.data)) * (
                                            gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (0))
                        ## w^2x^2
                        W_metric += -(gpts[name].r2) * (torch.abs(gpts[name].layer.weight.data) ** (2)) * (
                                gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (2))  # 768,128
                    # SVD
                    # U, S, VT = torch.linalg.svd(W_metric, full_matrices=False)
                    # self.logger.info(f"{name} W_metric svd S shape {S.shape}")

                    # calculate block outlier ratio
                    block_outlier_ratio = self.check_outlier_mean(torch.flatten(W_metric.cpu()),
                                                                  self.args.Hyper_m)
                    self.layer_outlier_ratios.append(block_outlier_ratio)
                    self.block_sizes.append(subset[name].weight.numel())
                total_params = sum(self.block_sizes)
                block_weights = np.array(self.block_sizes) / total_params
                self.all_blocks_ratio = np.array(self.layer_outlier_ratios)
                self.all_blocks_ratio = (self.all_blocks_ratio - self.all_blocks_ratio.min()) / (
                            self.all_blocks_ratio.max() - self.all_blocks_ratio.min())
                #  [target_sparsity - lambda, target_sparsity + lambda]
                target_sparsity = self.args.sparsity_ratio
                delta = (self.all_blocks_ratio - np.mean(self.all_blocks_ratio)) * self.args.Lambda * 2
                self.all_blocks_ratio = np.clip(target_sparsity + delta, 0.1, 0.95)

                # 3. weighted sparsity
                current_weighted_sparsity = np.sum(self.all_blocks_ratio * block_weights)
                scale = target_sparsity / current_weighted_sparsity
                self.all_blocks_ratio = 1 - np.clip(self.all_blocks_ratio * scale, 0.1, 0.95)

                self.logger.info(f"Block sparsity: {1 - self.all_blocks_ratio}, "
                                 f"Block outlier ratio: {self.all_blocks_ratio}, "
                                 f"Target sparsity: {target_sparsity:.4f}, "
                                 f"Weighted sparsity: {np.sum((1 - self.all_blocks_ratio) * block_weights):.4f}, ")
                self.logger.info("before layer sparsity compensation", self.layer_outlier_ratios)

                return self.all_blocks_ratio
            elif granularity == 'per-layer':
                self.layer_wmetric = []
                for name in subset:
                    # W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1, -1)))
                    if name in self.target_layer_names:
                        gpts = wrapped_layers
                    else:
                        gpts = gpts_layers
                    W_metric = (torch.abs(gpts[name].layer.weight.data) ** 2) * (
                            gpts[name].scaler_row.reshape((1, -1)) ** (1))
                    if self.args.d2_wanda:
                        # (lambda_1 ywx)^(1/2)
                        # W_metric += (self.r1 ** (1/2)) * (self.y_scaler_col.reshape((-1, 1)) ** (1 / 2)) * (torch.abs(self.layer.weight.data) ** (1 / 2)) * self.delta_x_scaler_row.reshape((1, -1)) ** (1/2)
                        # # (lambda_2 w^2x^2)^1/2=sqrt(lambda_2) wx
                        # W_metric += -(self.r2 ** (1/2)) * (torch.abs(self.layer.weight.data)) * (self.delta_x_scaler_row.reshape((1, -1)) ** (1))  # 768,128

                        # # new-->not scaling to sqrt: correct
                        ## ywx
                        W_metric += (gpts[name].r1) * (gpts[name].y_scaler_col.reshape((-1, 1)) ** (1)) * (
                            torch.abs(gpts[name].layer.weight.data)) * (
                                            gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (0))
                        ## w^2x^2
                        W_metric += -(gpts[name].r2) * (torch.abs(gpts[name].layer.weight.data) ** (2)) * (
                                gpts[name].delta_x_scaler_row.reshape((1, -1)) ** (2))  # 768,128
                    self.layer_wmetric.append(torch.flatten(W_metric.cpu()))
                self.layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in self.layer_wmetric])
                self.out_ratio_layer = self.check_outlier_mean(self.layer_wmetric, self.args.Hyper_m)
                return self.out_ratio_layer

    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use gpu device == embed_tokens.weight.device, if cpu, turn to gpu
        '''
        device = self.model.model.decoder.embed_tokens.weight.device  #
        if device.type == 'cpu':
            device = self.device
            self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions.to(self.device)
            self.model.decoder.final_layer_norm.to(self.device)
        else:
            device = device.index
        self.logger.info(f"using gpu to calibrate-->device: {device}")

        dtype = next(iter(self.model.parameters())).dtype  # torch.float16
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                           device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        for batch in train_loader:  #
            try:
                self.model(batch[0].reshape(-1, self.model.seq_len).to(device))  # batch[0]-->[1,2048]
            except ValueError:
                pass
        self.layers[layer_ind] = self.layers[layer_ind].module
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        self.model.config.use_cache = self.use_cache  # True
        if self.args.free:
            self.model.model.decoder.embed_tokens.to("cpu")
            self.model.model.decoder.embed_positions.to("cpu")
            self.model.decoder.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()
        return inps, outs, attention_mask, position_ids

    def forward_layer_wrapper(self, layer, inps, outs, attention_mask, position_ids):  # no position_ids for opt
        subset = self.find_layers(layer)
        gpts = {}
        wrapped_layers = {}
        for name in subset:
            if name not in self.target_layer_names:
                gpts[name] = D2SparseGPT(self.args, subset[name])
            else:
                wrapped_layers[name] = D2Wanda(self.args, subset[name])

        def add_batch_sparsegpt(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        def add_batch_wrapped_gpt(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles_sparsegpt = []
        handles_wrapped_gpt = []
        for name in subset:
            if name not in self.target_layer_names:
                handles_sparsegpt.append(subset[name].register_forward_hook(add_batch_sparsegpt(name)))
            else:
                handles_wrapped_gpt.append(subset[name].register_forward_hook(add_batch_wrapped_gpt(name)))
        for j in range(inps.shape[0]):
            with torch.no_grad():  # [1,2048,768]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]  # [1,2048,768)
        for h in handles_sparsegpt:
            h.remove()
        for h in handles_wrapped_gpt:
            h.remove()
        return subset, gpts, wrapped_layers

    @timeit
    def prune_layer_weight(self, subset, gpts, wrapped_layers):
        for i, name in enumerate(subset):
            if self.args.dsm != None:
                if self.args.granularity == 'per-block':
                    self.sparsity_ratio = 1 - self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    # self.sparsity_ratio = 1-self.all_blocks_ratio[i]
                    self.logger.info(
                        f"block sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
                elif self.args.granularity == 'per-layer':
                    self.sparsity_ratio = 1 - self.all_layers_ratio[int(self.index_layer.split('_')[-1])]
                    self.logger.info(
                        f"layer sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
            if name not in self.target_layer_names:  # update wights
                if self.d2_sparsegpt:
                    self.logger.info(f"pruning {name} by D2-SparseGPT: r1={self.args.r1}, r2={self.args.r2}")
                else:
                    self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()
            else:
                if self.d2_wanda:
                    self.logger.info(f"pruning {name} by D2-Wanda: r1={self.args.r1}, r2={self.args.r2}")
                else:
                    self.logger.info(f"pruning {name} by Wanda")
                wrapped_layers[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                wrapped_layers[name].free()
            torch.cuda.empty_cache()

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        inps, outs, attention_mask, position_ids = self.prepare_layer_calibration(train_loader)
        self.all_layers_ratio = []
        self.all_layers_blocks_ratio = []
        for i in trange(len(self.layers), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

            elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                dev = self.device
                layer.to(dev)
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

            start = time.time()
            # 1. forward layer wrapper
            subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids)

            # whether to prune by dynamic sparsity-->get subset layer sparsity-->
            if self.args.dsm != None:
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, self.args.dsm,
                                                                            self.args.granularity)
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    # self.logger.info(f"origin layer total sparsity ratio:{self.args.sparsity_ratio*len(self.all_blocks_ratio)}->adjustment layer {i} total sparsity ratio:{len(self.all_blocks_ratio)-self.all_blocks_ratio.sum()}")
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, wrapped_layers, self.args.dsm,
                                                                           self.args.granularity)
                    self.all_layers_ratio.append(self.out_ratio_layer)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                inps, outs = outs, inps

            if not self.all_layers_ratio and not self.all_layers_blocks_ratio:
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
            del layer, subset, gpts
            gc.collect()
            torch.cuda.empty_cache()
            if self.args.free:
                self.layers[i].to("cpu")
                torch.cuda.empty_cache()
        # print(self.all_layers_ratio, self.all_layers_blocks_ratio)
        if self.all_layers_ratio or self.all_layers_blocks_ratio:
            print(self.all_layers_blocks_ratio)
            inps, outs, attention_mask, position_ids = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer':
                # self.logger.info(self.all_layers_ratio)
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                        1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (
                        1 - self.args.sparsity_ratio)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                    layer.to(dev)
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
                start = time.time()
                # 1. forward layer wrapper
                subset, gpts, wrapped_layers = self.forward_layer_wrapper(layer, inps, outs, attention_mask,
                                                                          position_ids)

                # 2. pruning weight
                self.prune_layer_weight(subset, gpts, wrapped_layers)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps

                del layer, subset, gpts, wrapped_layers
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
        self.model.config.use_cache = self.use_cache
        torch.cuda.empty_cache()
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")


class Prune_OPT:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.nsamples = args.nsamples
        self.device = args.device

        self.sparsity_ratio = args.sparsity_ratio
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = args.logger

    def init_model(self):  # share
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.decoder.layers

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-6):
        self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                # count += (W==0).sum().item()
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                # sub_count += (W == 0).sum().item()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count) / total_params

    @staticmethod
    def check_outlier_mean(mask, threshold):
        W = mask
        count = 0
        total_params = 0
        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio

    @staticmethod
    def compute_sensitivity(S, top_ratio=0.05, mode="clamp"):
        if mode == "abs":
            S = torch.abs(S)
        elif mode == "clamp":
            S = torch.clamp(S, min=0.0)

        if S.sum() == 0:
            return 0.0

        S_sorted, _ = torch.sort(S, descending=True)
        k = max(1, int(len(S) * top_ratio))
        topk_energy = S_sorted[:k].sum()
        total_energy = S_sorted.sum()

        return (topk_energy / total_energy).item()

    @torch.no_grad()
    def get_layer_dynamic_sparsity(self, subset, gpts, dsm='owl', granularity='per-block'):
        """
        Sparsity compensation
        Compensate for over-pruning caused by uniform sparsity due to different layer sensitivities, and balance sparsity.
        :param dsm:dynamic sparsity method-->global static adjustments
        :return:subset each layer sparsity
        """
        if dsm == "owl":
            # self.layer_wmetric = []
            # self.layer_outlier_ratios = {}
            if granularity == 'per-block':
                self.layer_outlier_ratios = []
                self.block_sizes = []
                self.block_importance = []
                for name in subset:
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(gpts[name].scaler_row.reshape((1, -1)))
                    # SVD
                    # U, S, VT = torch.linalg.svd(W_metric, full_matrices=False)
                    # self.logger.info(f"{name} W_metric svd U shape {U.shape}")
                    # self.logger.info(f"{name} W_metric svd S shape {S.shape}")
                    # self.logger.info(f"{name} W_metric svd VT shape {VT.shape}")
                    # self.logger.info(f"{name} S MIN {S.min()}, MAX {S.max()}, MEAN {S.mean()}, MEDIAN {S.median()}")

                    # # calculate block outlier ratio
                    block_outlier_ratio = self.check_outlier_mean(torch.flatten(W_metric.cpu()),
                                                                  self.args.Hyper_m)  # why each has the same Hyper_m
                    # block_outlier_ratio = self.compute_sensitivity(S, top_ratio=0.2, mode="abs")

                    # block_outlier_ratio = self.check_outlier_mean(torch.flatten(S.cpu()), self.args.Hyper_m) #
                    self.layer_outlier_ratios.append(block_outlier_ratio)
                    self.block_sizes.append(subset[name].weight.numel())
                total_params = sum(self.block_sizes)
                block_weights = np.array(self.block_sizes) / total_params
                self.all_blocks_ratio = np.array(self.layer_outlier_ratios)
                self.all_blocks_ratio = (self.all_blocks_ratio - self.all_blocks_ratio.min()) / (
                            self.all_blocks_ratio.max() - self.all_blocks_ratio.min())
                #  [target_sparsity - lambda, target_sparsity + lambda]
                target_sparsity = self.args.sparsity_ratio
                delta = (self.all_blocks_ratio - np.mean(self.all_blocks_ratio)) * self.args.Lambda * 2
                self.all_blocks_ratio = np.clip(target_sparsity + delta, 0.1, 0.95)

                # 3. weighted sparsity
                current_weighted_sparsity = np.sum(self.all_blocks_ratio * block_weights)
                scale = target_sparsity / current_weighted_sparsity
                self.all_blocks_ratio = 1 - np.clip(self.all_blocks_ratio * scale, 0.1, 0.95)

                self.logger.info(f"Block sparsity: {1 - self.all_blocks_ratio}, "
                                 f"Block outlier ratio: {self.all_blocks_ratio}, "
                                 f"Target sparsity: {target_sparsity:.4f}, "
                                 f"Weighted sparsity: {np.sum((1 - self.all_blocks_ratio) * block_weights):.4f}, ")
                return self.all_blocks_ratio
            elif granularity == 'per-layer':
                self.layer_wmetric = []
                for name in subset:
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(gpts[name].scaler_row.reshape((1, -1)))
                    self.layer_wmetric.append(torch.flatten(W_metric.cpu()))
                self.layer_wmetric = torch.cat(
                    [torch.flatten(x.cpu()) for x in self.layer_wmetric])  # [total weight number]
                self.out_ratio_layer = self.check_outlier_mean(self.layer_wmetric, self.args.Hyper_m)
                return self.out_ratio_layer

    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use gpu device == embed_tokens.weight.device, if cpu, turn to gpu
        '''
        device = self.model.model.decoder.embed_tokens.weight.device  #
        if device.type == 'cpu':
            device = self.device
            self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions.to(self.device)
            self.model.decoder.final_layer_norm.to(self.device)
        else:
            device = device.index
        self.logger.info(f"using gpu to calibrate-->device: {device}")

        dtype = next(iter(self.model.parameters())).dtype  # torch.float16
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                           device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        for batch in train_loader:  #
            try:
                self.model(batch[0].reshape(-1, self.model.seq_len).to(device))  # batch[0]-->[1,2048]
            except ValueError:
                pass
        self.layers[layer_ind] = self.layers[layer_ind].module
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        self.model.config.use_cache = self.use_cache  # True
        if self.args.free:
            self.model.model.decoder.embed_tokens.to("cpu")
            self.model.model.decoder.embed_positions.to("cpu")
            self.model.decoder.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()
        return inps, outs, attention_mask, position_ids

    def forward_layer_wrapper(self, layer, inps, outs, attention_mask, position_ids, GPT):
        subset = self.find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = GPT(self.args, subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(inps.shape[0]):
            with torch.no_grad():  # [1,2048,768]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]  # [1,2048,768)
        for h in handles:
            h.remove()
        return subset, gpts

    @timeit
    def prune_layer_weight(self, subset, gpts):
        for i, name in enumerate(subset):
            if self.args.dsm != None:
                if self.args.granularity == 'per-block':
                    # self.sparsity_ratio = 1-self.all_blocks_ratio[i]
                    self.sparsity_ratio = 1 - self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    # self.sparsity_ratio = self.all_layers_blocks_ratio[int(self.index_layer.split('_')[-1])][i]
                    self.logger.info(
                        f"block sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
                elif self.args.granularity == 'per-layer':
                    self.sparsity_ratio = 1 - self.all_layers_ratio[int(self.index_layer.split('_')[-1])]
                    self.logger.info(
                        f"layer sparsity  compensate, origin sparsity:{self.args.sparsity_ratio}->new sparsity:{self.sparsity_ratio}")
            if self.args.prune_method == 'sparsegpt':
                self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m,
                                       blocksize=128, percdamp=.01)
                gpts[name].free()

            elif self.args.prune_method == 'wanda':
                self.logger.info(f"pruning {name} by Wanda")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()

            elif self.args.prune_method == 'pruner-zero':
                self.logger.info(f"pruning {name} by Pruner-Zero")
                indexed_name = f'{name}_{self.index_layer}'
                gradients = self.gradients_l2[indexed_name]
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, gradients, engine=self.engine)
                gpts[name].free()

            else:
                raise NotImplementedError
            torch.cuda.empty_cache()

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        inps, outs, attention_mask, position_ids = self.prepare_layer_calibration(train_loader)
        if self.args.prune_method == 'pruner-zero':
            self.logger.info("you must loading model gradient for pruner-zero")
            self.gradients_l2 = self.args.gradients_l2
            self.engine = self.args.engine  # GPTree.load_tree('../Pruner-Zero/data/best_tree.json')
        self.all_layers_ratio = []
        self.all_layers_blocks_ratio = []
        for i in trange(len(self.layers), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

            elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                dev = self.device
                layer.to(dev)
                inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

            start = time.time()
            # 1. forward layer wrapper
            if self.args.prune_method == 'sparsegpt':
                GPT = SparseGPT
            elif self.args.prune_method == 'wanda':
                GPT = Wanda
            elif self.args.prune_method == 'pruner-zero':
                GPT = PrunerZero
            else:
                raise NotImplementedError

            # 1. forward layer wrapper
            subset, gpts = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, GPT)
            # whether to prune by dynamic sparsity-->get subset layer sparsity-->
            if self.args.dsm != None:
                if self.args.granularity == 'per-block':
                    self.all_blocks_ratio = self.get_layer_dynamic_sparsity(subset, gpts, self.args.dsm,
                                                                            self.args.granularity)
                    self.logger.info(f'layer {i} blocks outlier ratio{self.all_blocks_ratio}')
                    # self.logger.info(self.all_blocks_ratio, np.mean(self.all_blocks_ratio), np.max(self.all_blocks_ratio), np.min(self.all_blocks_ratio))
                    # self.logger.info(
                    #     f"origin layer total sparsity ratio:{self.args.sparsity_ratio * len(self.all_blocks_ratio)}->adjustment layer {i} total sparsity ratio:{len(self.all_blocks_ratio) - self.all_blocks_ratio.sum()}")
                    self.all_layers_blocks_ratio.append(self.all_blocks_ratio)
                elif self.args.granularity == 'per-layer':
                    self.out_ratio_layer = self.get_layer_dynamic_sparsity(subset, gpts, self.args.dsm,
                                                                           self.args.granularity)
                    self.all_layers_ratio.append(self.out_ratio_layer)
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                inps, outs = outs, inps
            if not self.all_layers_ratio and not self.all_layers_blocks_ratio:
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
            del layer, subset, gpts
            gc.collect()
            torch.cuda.empty_cache()
            if self.args.free:
                self.layers[i].to("cpu")
                torch.cuda.empty_cache()
        if self.all_layers_ratio or self.all_layers_blocks_ratio:
            inps, outs, attention_mask, position_ids = self.prepare_layer_calibration(train_loader)
            if self.args.granularity == 'per-layer':
                self.all_layers_ratio = np.array(self.all_layers_ratio)
                self.all_layers_ratio = ((self.all_layers_ratio - self.all_layers_ratio.min()) * (
                        1 / (self.all_layers_ratio.max() - self.all_layers_ratio.min()) * self.args.Lambda * 2))
                self.all_layers_ratio = self.all_layers_ratio - np.mean(self.all_layers_ratio) + (
                            1 - self.args.sparsity_ratio)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                self.index_layer = f'layer_{i}'
                if f"model.layers.{i}" in self.model.hf_device_map:  # multiple gpu can run, this means model init by "auto"
                    dev = self.model.hf_device_map[f"model.layers.{i}"]
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
                elif layer.self_attn.q_proj.weight.device.type == 'cpu':  # single gpu can run, running by offload
                    dev = self.device
                    layer.to(dev)
                    inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)
                start = time.time()
                # 1. forward layer wrapper
                if self.args.prune_method == 'sparsegpt':
                    GPT = SparseGPT
                elif self.args.prune_method == 'wanda':
                    GPT = Wanda
                elif self.args.prune_method == 'pruner-zero':
                    GPT = PrunerZero
                else:
                    raise NotImplementedError
                # 1. forward layer wrapper
                subset, gpts = self.forward_layer_wrapper(layer, inps, outs, attention_mask, position_ids, GPT)
                # 2. pruning weight
                self.prune_layer_weight(subset, gpts)
                # 3. forward layers
                for j in range(self.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                # update next layer inputs
                self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
                inps, outs = outs, inps
                del layer, subset, gpts
                gc.collect()
                torch.cuda.empty_cache()
                if self.args.free:
                    self.layers[i].to("cpu")
                    torch.cuda.empty_cache()
        self.model.config.use_cache = self.use_cache
        torch.cuda.empty_cache()
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")








