

import argparse
import ast

class Prune_Args:
    def __init__(self, cfg):
        '''
        :param cfg: -->dict-->example:{model: llama-2-13b, nsamples: 128}
        '''
        self.cfg = cfg
        self.parser = argparse.ArgumentParser()
        self._gen_args()
        self.args = self.parser.parse_args()
        self.args.target_layer_names = ast.literal_eval(self.args.target_layer_names)
        self.args.tasks = ast.literal_eval(self.args.tasks)

    def _gen_args(self):
        self.parser.add_argument('--model', type=str, help='path to pre-trained llm directory, i.e. llama-2-13b', default=self.cfg["model"])
        self.parser.add_argument('--exp_name', type=str, help='experiment name', default=self.cfg["exp_name"])
        self.parser.add_argument("--cali_dataset", default=self.cfg['cali_dataset'],
                                 type=str, help="calibration dataset")
        self.parser.add_argument('--cali_data_path', type=str, help='calibration data path', default=self.cfg["cali_data_path"])
        self.parser.add_argument("--eval_dataset", default=self.cfg['eval_dataset'],
                                 type=str, help="calibration dataset")
        self.parser.add_argument('--eval_data_path', type=str, help='eval data path', default=self.cfg["eval_data_path"])
        self.parser.add_argument("--data_cache_dir", default=self.cfg['data_cache_dir'], type=str, help="processed cali/eval data cache dir")
        # self.parser.add_argument('--log_dir', type=str, help='log dir path', default=self.cfg["log_dir"])
        self.parser.add_argument('--output_dir', type=str, help='output dir path', default=self.cfg["output_dir"])
        self.parser.add_argument('--seed', type=int, default=self.cfg['seed'],
                                 help='Seed for sampling the calibration data.')
        self.parser.add_argument('--nsamples', type=int, default=self.cfg['nsamples'],
                                 help='Number of calibration samples.')  # 128 default
        self.parser.add_argument('--prune_m', type=int, default=self.cfg['prune_m'],
                                 help='parameter m of n:m pruning')  #
        self.parser.add_argument('--prune_n', type=int, default=self.cfg['prune_n'],
                                 help='parameter n of n:m pruning')  #
        self.parser.add_argument('--percdamp', type=float, default=.01,
                            help='Percent of the average Hessian diagonal to use for dampening.')

        self.parser.add_argument('--sparsity_ratio', type=float, default=self.cfg['sparsity_ratio'], help='Sparsity level')
        self.parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "3:4"],
                                 default=self.cfg['sparsity_type'])
        self.parser.add_argument("--prune_method",
                                 type=str,
                                 choices=["magnitude", "wanda",
                                          "sparsegpt", "ablate_mag_seq",
                                          "ablate_wanda_seq", "ablate_mag_iter",
                                          "ablate_wanda_iter", "search", "pruner-zero", "sparsellm", "d2prune"],
                                 default=self.cfg['prune_method'])

        self.parser.add_argument("--cache_dir", default=self.cfg['cache_dir'], type=str, help="cache dir")

        self.parser.add_argument('--use_variant', action="store_true",
                                 help="whether to use the wanda variant described in the appendix")

        self.parser.add_argument('--save_model', type=str, default=self.cfg['save_model'],
                                 help='Path to save the pruned model.')

        self.parser.add_argument("--eval_zero_shot", action="store_true")
        self.parser.add_argument("--eval_ppl", action="store_true")
        self.parser.add_argument("--test_offload", action="store_true", help="whether to offload memory to cpu")

        self.parser.add_argument("--device", type=str, default=self.cfg['device'], help="Device to use for calibration")
        self.parser.add_argument('--kmeans', action="store_true")
        self.parser.add_argument('--s', type=float, default=self.cfg['s'], help='activation manitude')
        self.parser.add_argument('--auto_s', action="store_true", help='model seq len for auto s')
        self.parser.add_argument('--r1', type=float, default=self.cfg['r1'], help='First-order activation bias term coefficient 1, i.e., $\lambda_1$ ywx')
        self.parser.add_argument('--r2', type=float, default=self.cfg['r2'], help='Second-order activation bias term coefficient 2, i.e, $\lambda_2$ x^tww^tx')
        self.parser.add_argument('--d2_wanda', action="store_true")
        self.parser.add_argument('--d2_sparsegpt', action="store_true")
        self.parser.add_argument('--EA', action="store_true", help="Exponential adaptation/adjustment for activations ||X||^R or ||Y||^R, R=[0, 1/2, 1, 2]")
        self.parser.add_argument('--free', action="store_true")
        self.parser.add_argument('--distribute', action="store_true")
        # self.parser.add_argument('--blocksize', type=int, default=self.cfg['blocksize'], help='sparsegpt block')

        self.parser.add_argument('--target_layer_names', type=str,
                                 default=self.cfg['target_layer_names'],
                                 help='which layer to prune without weights update')
        self.parser.add_argument('--tasks', type=str, default=self.cfg['tasks'], help='zero-shot tasks')
        self.parser.add_argument('--dsm', type=str, default=None, choices=['owl', 'besa', 'evopress', 'als', 'dsa'],
                                 help="dynamic layer sparsity method")
        self.parser.add_argument('--granularity', type=str, default='per-block', choices=['per-block', 'per-layer'],
                                 help="dynamic layer sparsity method")
        self.parser.add_argument(
            "--Lambda",
            default=0.08,
            type=float,
            help="Lambda for owl",
        )
        self.parser.add_argument(
            "--Hyper_m",
            type=float,
            default=3,
            help="Hyper_m for owl",
        )






