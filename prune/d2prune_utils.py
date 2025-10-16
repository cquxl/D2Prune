"""layer wrapper for handling the result of each layer, such as Hessian matrix or ||x||2"""

import math
import time
import torch
import torch.nn as nn
import transformers
from sklearn.cluster import KMeans
from torch import Tensor


class D2SparseGPT:
    '''
    ## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    Add the 1st-order activation derivatives based on the Sparsegpt
    '''
    def __init__(self, args, layer): # layer: torch.nn.Module
        self.args = args
        self.layer = layer

        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        if self.args.d2_sparsegpt:
            self.y_scaler_col = torch.zeros((self.rows), device=self.dev)
            self.delta_x_scaler_row = torch.zeros((self.columns), device=self.dev)

        if self.args.dsm !=None:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)

        self.nsamples = 0

        self.s = self.args.seq_len if self.args.auto_s else self.args.s
        self.r1 = self.args.r1
        self.r2 = self.args.r2

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)

        if self.args.d2_sparsegpt:
            self.y_scaler_col *= self.nsamples / (self.nsamples + tmp)
            self.delta_x_scaler_row *= self.nsamples / (self.nsamples + tmp)
        if self.args.dsm !=None:
            self.scaler_row *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

        if self.args.d2_sparsegpt:
            if self.args.kmeans:
                self.y_scaler_col += torch.norm(out.reshape(-1, out.shape[-1]), p=2, dim=0)
                self.delta_x_scaler_row += torch.norm(inp, p=2, dim=1) ** 2
                self.s = self.cal_s_by_kmeans()
                self.y_scaler_col /= self.s
                self.delta_x_scaler_row /= self.s
            else:
                self.y_scaler_col += torch.norm(out.reshape(-1, out.shape[-1]), p=2, dim=0) / self.s # y
                self.delta_x_scaler_row += torch.norm(inp, p=2, dim=1) / self.s  # x

        if self.args.dsm != None:
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples # ||x||^2


    def cal_s_by_kmeans(self) -> Tensor:
        '''
        this function is used to calculate the activation scaling factor s by kmeans automatically
        '''
        class_pred = KMeans(n_clusters=2, random_state=self.args.seed).fit_predict(self.y_scaler_col.cpu().numpy().reshape(-1, 1))
        s = (self.y_scaler_col[class_pred == 0] ** 2).mean() if (self.y_scaler_col[class_pred == 1] ** 2).mean() > (
                    self.y_scaler_col[class_pred == 0] ** 2).mean() else (
                    self.y_scaler_col[class_pred == 0] ** 2).mean()
        return s

    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Losses = torch.zeros(self.rows, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        mask = None
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)  # [768,128]
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]  # [128,128]
            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    if self.args.d2_sparsegpt:  # w * y * delta_x, x^t*w*w^t*delta_x

                        tmp += self.r1 * ((self.y_scaler_col.reshape((-1, 1)) ** (1)) * torch.abs(W1) * (
                                    self.delta_x_scaler_row[i1:i2].reshape((1, -1)) ** (1))) # ywx

                        tmp += -self.r2 * (W1 ** 2) *(self.delta_x_scaler_row[i1:i2].reshape((1, -1)) ** (2)) # 768,128 w^2x^2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if prune_n != 0 and i % prune_m == 0:
                    self.args.logger.info(f"pruning N:M--> {prune_n}:{prune_m}")
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    if self.args.d2_sparsegpt:
                        tmp += self.r1 * ((self.y_scaler_col.reshape((-1, 1)) ** (1)) * torch.abs(W1[:, i:(i + prune_m)]) * (
                                self.delta_x_scaler_row[i1:i2][i:(i + prune_m)].reshape((1, -1)) ** (1)))  # lambda_1 * ywx
                        # lambda_2 w^2x^2
                        tmp += -self.r2 * (W1[:, i:(i + prune_m)] ** 2) * (self.delta_x_scaler_row[i1:i2][i:(i + prune_m)].reshape((1, -1)) ** (2))  # 768,128
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))  # [768,i:]
                Err1[:, i] = err1
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        if self.args.d2_sparsegpt:
            self.y_scaler_col = None
            self.delta_x_scaler_row = None
        torch.cuda.empty_cache()


class D2Wanda:
    '''
    ## Wanda: https://github.com/locuslab/wanda
    Add the 1st and 2nd-order activation derivatives based on the Wanda
    '''
    def __init__(self, args, layer):
        self.args = args
        self.layer = layer
        self.dev = self.layer.weight.device

        self.rows = layer.weight.data.shape[0]  # 3072
        self.columns = layer.weight.data.shape[1]  # 768

        self.scaler_row = torch.zeros((self.columns), device=self.dev)  # [768]
        if self.args.d2_wanda:
            self.y_scaler_col = torch.zeros((self.rows), device=self.dev)
            self.delta_x_scaler_row = torch.zeros((self.columns), device=self.dev)

        self.nsamples = 0
        self.s = self.args.seq_len if self.args.auto_s else self.args.s
        self.r1 = self.args.r1
        self.r2 = self.args.r2

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # [768,2048]

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        if self.args.d2_wanda:
            self.y_scaler_col *= self.nsamples / (self.nsamples + tmp)
            self.delta_x_scaler_row *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples # ||x||^2


        if self.args.d2_wanda:
            if self.args.kmeans:
                self.y_scaler_col += torch.norm(out.reshape(-1, out.shape[-1]), p=2, dim=0)
                self.delta_x_scaler_row += torch.norm(inp, p=2, dim=1)
                self.s = self.cal_s_by_kmeans()
                self.y_scaler_col /= self.s
                self.delta_x_scaler_row /= self.s
            else:
                self.y_scaler_col += torch.norm(out.reshape(-1, out.shape[-1]), p=2, dim=0) / self.s # y
                self.delta_x_scaler_row += torch.norm(inp, p=2, dim=1) / self.s  # x

    def cal_s_by_kmeans(self) -> Tensor:
        '''
        this function is used to calculate the activation scaling factor s by kmeans automatically
        '''
        class_pred = KMeans(n_clusters=2, random_state=self.args.seed).fit_predict(self.y_scaler_col.cpu().numpy().reshape(-1, 1))
        s = (self.y_scaler_col[class_pred == 0]**2).mean() if (self.y_scaler_col[class_pred == 1]**2).mean() > (self.y_scaler_col[class_pred == 0]**2).mean() else (self.y_scaler_col[class_pred == 0]**2).mean()
        # self.args.logger.info(f"scaling factor s by kmeans for output activations: {s}")
        return s

    def cal_s_by_mean_outliers(self):
        pass



    def fasterprune(self, sparsity, prune_n=0, prune_m=0):
        # org-->scaling to sqrt, Cauchy-Schwarz inequality
        # W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1, -1)))
        W_metric = (torch.abs(self.layer.weight.data) ** 2) * (self.scaler_row.reshape((1, -1)) ** (1))  # w^2 * x^2
        if self.args.d2_wanda:
            # (lambda_1 ywx)^(1/2)
            # W_metric += (self.r1 ** (1/2)) * (self.y_scaler_col.reshape((-1, 1)) ** (1 / 2)) * (torch.abs(self.layer.weight.data) ** (1 / 2)) * self.delta_x_scaler_row.reshape((1, -1)) ** (1/2)
            # # (lambda_2 w^2x^2)^1/2=sqrt(lambda_2) wx
            # W_metric += -(self.r2 ** (1/2)) * (torch.abs(self.layer.weight.data)) * (self.delta_x_scaler_row.reshape((1, -1)) ** (1))  # 768,128

            # # new-->not scaling to sqrt: correct
            ## ywx
            W_metric += (self.r1) * (self.y_scaler_col.reshape((-1, 1)) ** (1)) * (torch.abs(self.layer.weight.data)) * (self.delta_x_scaler_row.reshape((1, -1)) ** (1))
            ## w^2x^2
            W_metric += -(self.r2) * (torch.abs(self.layer.weight.data) ** (2))  * (self.delta_x_scaler_row.reshape((1, -1)) ** (2))  # 768,128

        W_mask = (torch.zeros_like(W_metric) == 1)

        if prune_n != 0:
            self.args.logger.info(f"pruning N:M--> {prune_n}:{prune_m}")
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)
        self.layer.weight.data[W_mask] = 0
        W_metric = None
        W_mask = None
        torch.cuda.empty_cache()

    def free(self):
        self.scaler_row = None
        if self.args.d2_wanda:
            self.y_scaler_col = None
            self.delta_x_scaler_row = None
        torch.cuda.empty_cache()


