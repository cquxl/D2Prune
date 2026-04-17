# Pruning Examples

sh scripts/opt-125m.sh

sh scripts/llama-2-7b.sh

sh scripts/llama-2-13b.sh

sh scripts/llama-2-70b.sh

sh scripts/llama-3-8b.sh

# D²Prune: Sparsifying Large Language Models via Dual Taylor Expansion and Attention Distribution Awareness

[![Code](https://img.shields.io/badge/Code-Open-4CAF50.svg)](https://anonymous.4open.science/r/D2Prune-FF87)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![1748418085466](image/README/1748418085466.png)

## 🚀 Key Innovations

D²Prune is an efficient post-training pruning method for large language models (LLMs), addressing two critical limitations of traditional approaches at high sparsity:

1. **Activation distribution shift**: Ignoring differences between calibration and test data leads to inaccurate error estimation.
2. **Destruction of attention long-tail distribution**: Failing to preserve the importance of key tokens in attention modules degrades reasoning capabilities.

### Core Methods

- **Dual Taylor Expansion Pruning Mechanism**  By modeling second-order perturbations of both weights ((W)) and activations ((X)), we derive a dual Taylor expansion of the error function:

$$
\delta E=\lambda _1y\boldsymbol{w}^T\boldsymbol{x}+\lambda _2\boldsymbol{x}^T\boldsymbol{H}_{22}\boldsymbol{x}+\frac{1}{2}\delta \boldsymbol{w}^T\boldsymbol{H}_{11}\delta \boldsymbol{w}
$$

This captures the interaction between weights and activations to improve pruning mask selection.

- **Attention Distribution-Aware Dynamic Update Strategy**
  Treating the update states of Q/K/V weights as a combinatorial optimization problem, we dynamically select optimal configurations by minimizing perplexity (PPL), preserving the long-tail attention distribution:

$$
\mathrm{arg}\min_{\delta \boldsymbol{w}_{q/k/v}} \;\,L_{q/k/v}+\rho \,\mathrm{D}_{\mathrm{KL}}\!\bigl( \mathrm{MHA(}\tilde{\boldsymbol{w}}_{q/k/v})_i\left\| \mathrm{MHA(}\boldsymbol{w}_{q/k/v})_i \right. \bigr)
$$

This reduces the root-mean-square error (RMSE) of attention layers by 61.2% on average.

## 💡 Main Advantages

| Dimension                | D²Prune Features                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Accuracy**       | Achieves 8.2% higher zero-shot accuracy than SparseGPT on LLaMA-2-7B at 80% sparsity.                  |
| **Efficiency**     | Post-training pruning requires no retraining, with 1.3× inference speedup (2:4 sparsity).             |
| **Generalization** | Works on OPT, LLaMA, Qwen3, and DeiT, outperforming baselines across architectures.                    |
| **Robustness**     | Reduces perplexity by 15% compared to Wanda on tasks with large distribution shifts (e.g., HellaSwag). |

## 📊 Key Experimental Results

### Language Modeling Performance (WikiText2 Perplexity)

| Sparsity      | Method             | OPT-125M          | LLaMA-2-7B       | LLaMA-2-13B      | LLaMA-2-70B     | LLaMA-3-8B       |
| ------------- | ------------------ | ----------------- | ---------------- | ---------------- | --------------- | ---------------- |
| 0             | Dense              | 27.66             | 5.12             | 4.57             | 3.12            | 5.54             |
| **50**  | SparseGPT          | 36.85             | 6.52             | 5.63             | 3.98            | 8.56             |
|               | Wanda              | 38.88             | 6.44             | 5.58             | 3.98            | 9.06             |
|               | Pruner-Zero        | 38.80             | 6.43             | 5.57             | \-              | 8.52             |
|               | **D²Prune** | **34.98**   | **6.36**   | **5.53**   | **3.93**  | **8.34**   |
| **60**  | SparseGPT          | 59.46             | 9.56             | 7.77             | 4.98            | 14.40            |
|               | Wanda              | 74.39             | 9.89             | 7.87             | 4.99            | 22.80            |
|               | Pruner-Zero        | 68.07             | 10.34            | 7.82             | \-              | 20.30            |
|               | **D²Prune** | **52.10**   | **9.05**   | **7.49**   | **4.88**  | **13.44**  |
| **70**  | SparseGPT          | 218.29            | 29.62            | 18.20            | 8.61            | 38.85            |
|               | Wanda              | 347.42            | 84.92            | 44.86            | 40.27           | 114.30           |
|               | Pruner-Zero        | 317.87            | 151.92           | 44.76            | \-              | 280.33           |
|               | **D²Prune** | **160.81**  | **21.10**  | **16.51**  | **8.17**  | **33.37**  |
| **80**  | SparseGPT          | 2140.55           | 102.43           | 99.14            | 25.86           | 178.01           |
|               | Wanda              | 1920.63           | 5107.20          | 1384.40          | 156.68          | 2245.91          |
|               | Pruner-Zero        | 1251.38           | 10244.70         | 2040.65          | \-              | 10420.01         |
|               | **D²Prune** | **1038.87** | **92.68**  | **76.80**  | **21.37** | **151.47** |
| **2:4** | SparseGPT          | 59.76             | 10.18            | 8.39             | 5.32            | 14.16            |
|               | Wanda              | 79.80             | 11.35            | 8.37             | 5.18            | 22.86            |
|               | Pruner-Zero        | 70.92             | 11.16            | **8.02**   | \-              | 23.56            |
|               | **D²Prune** | **59.43**   | **10.00**  | 8.05             | **5.12**  | **14.10**  |
| **3:4** | SparseGPT          | 1365.58           | 154.23           | 147.23           | 54.84           | 281.74           |
|               | Wanda              | 2497.68           | 3111.14          | 5815.71          | 386.57          | 13054.17         |
|               | Pruner-Zero        | 2946.15           | 7913.18          | 4134.56          | \-              | 854085.30        |
|               | **D²Prune** | **1346.36** | **136.89** | **143.69** | **48.06** | **190.35** |

### Zero-Shot Task Accuracy (7-Task Average)

| Sparsity     | Method             | OPT-125M        | LLaMA-2-7B      | LLaMA-2-13B     | LLaMA-2-70B     | LLaMA-3-8B      |
| ------------ | ------------------ | --------------- | --------------- | --------------- | --------------- | --------------- |
| 0            | Dense              | 39.68           | 64.38           | 67.06           | 71.50           | 68.40           |
| **50** | SparseGPT          | 39.82           | 60.35           | 64.87           | 71.35           | 63.13           |
|              | Wanda              | 39.75           | 60.46           | 64.17           | 70.96           | 61.02           |
|              | Pruner-Zero        | 39.09           | 59.26           | 64.21           | \-              | 60.28           |
|              | **D²Prune** | **40.39** | **61.06** | **65.90** | **71.60** | **63.58** |
| **60** | SparseGPT          | 39.37           | 55.34           | 60.26           | 70.09           | 55.39           |
|              | Wanda              | 39.45           | 53.91           | 59.57           | 69.04           | 48.12           |
|              | Pruner-Zero        | 39.32           | 52.52           | 58.12           | \-              | 50.65           |
|              | **D²Prune** | **40.14** | **56.08** | **60.81** | **70.65** | **56.95** |
| **70** | SparseGPT          | 36.31           | 44.75           | 48.12           | 63.49           | 43.56           |
|              | Wanda              | 35.43           | 36.68           | 39.75           | 60.44           | 37.07           |
|              | Pruner-Zero        | 36.88           | 38.09           | 43.33           | \-              | 36.17           |
|              | **D²Prune** | **38.09** | **47.97** | **48.32** | **64.06** | **44.82** |
| **80** | SparseGPT          | 35.08           | 36.23           | 38.28           | 47.71           | 36.95           |
|              | Wanda              | 34.93           | 33.72           | 34.67           | 37.98           | 34.87           |
|              | Pruner-Zero        | 35.06           | 34.83           | 35.09           | \-              | 35.31           |
|              | **D²Prune** | **36.29** | **39.09** | **39.42** | **48.09** | **38.73** |

## 📝 Preparations

### 1. Install Dependencies

```bash
# create environment
conda create -n d2prune python=3.10
conda activate d2prune
# install torch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# install other packages (transformers==4.37.0, lm-eval==0.4.4)
git clone https://github.com/20250516aaa/D2Prune.git
cd D2Prune
pip install -r requirements.txt
```

### 2. Data Preparation

datasets and models can be download from Huggingface: [https://huggingface.co/datasets](https://huggingface.co/datasets)

* Download datasets in your local directory, e.g., '../cache/data'

**Language Model for PPL evaluation**

| Type        | Datasets  | Local dir              | url                                                                                   |
| ----------- | --------- | ---------------------- | ------------------------------------------------------------------------------------- |
| calibration | C4        | ../cache/data/c4       | [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)     |
| evaluation  | WikiText2 | ../cache/data/wikitext | [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4) |

**Zero-shot for Accuracy evaluation**

| Datasets         | Local Dir                                | Url                                                                                                   |
| ---------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| BoolQ            | ~/.cache/huggingface/datasets/boolq      | [https://huggingface.co/datasets/google/boolq](https://huggingface.co/datasets/google/boolq)             |
| HellaSwag        | ~/.cache/huggingface/datasets/hellaswag  | [https://huggingface.co/datasets/hellaswag](https://huggingface.co/datasets/hellaswag)                   |
| WinoGrande       | ~/.cache/huggingface/datasets/winogrande | [https://huggingface.co/datasets/winogrande](https://huggingface.co/datasets/winogrande)                 |
| RTE              | ~/.cache/huggingface/datasets/glue       | [https://huggingface.co/datasets/nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue)             |
| ARC-c and ARC-e | ~/.cache/huggingface/datasets/ai2_arc    | [https://huggingface.co/datasets/ai2_arc](https://huggingface.co/datasets/ai2_arc)                       |
| OBQA             | ~/.cache/huggingface/datasets/openbookqa | [https://huggingface.co/datasets/allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) |

### 3. Model Preparation

models can be download from Huggingface: [https://huggingface.co/](https://huggingface.co/datasets)

Download model weights  in your local directory, e.g. '../cache/llm_weights'

| Models      | Local Dir                                  | Url                                                                                                         |
| ----------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| OPT-125M    | ../cache/llm_weights/opt-125m              | [https://huggingface.co/facebook/opt-125m](https://huggingface.co/facebook/opt-125m)                           |
| LLaMA-2-7B  | ../cache/llm_weights/llama-2-7b            | [https://huggingface.co/meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)                   |
| LLaMA-2-13B | ../cache/llm_weights/llama-2-13b           | [https://huggingface.co/meta-llama/Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b)                 |
| LLaMA-2-70B | ../cache/llm_weights/llama-2-70b           | [https://huggingface.co/meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)                 |
| LLaMA-3-8B  | ../cache/llm_weights/llama-3-8b            | [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)         |
| Qwen3-8B    | ../cache/llm_weights/qwen3-8b              | [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                   |
| Qwen3-14B   | ../cache/llm_weights/qwen3-14b             | [https://huggingface.co/Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)                                 |
| DeiT        | ../cache/llm_weights/deit-base-patch16-224 | [https://huggingface.co/facebook/deit-base-patch16-224](https://huggingface.co/facebook/deit-base-patch16-224) |

## 🛠️ Usage

We provide full script to run D2Prune in ./scripts/. Here, we use opt-125m as an example.

1. For OPT-125M

**unstructured pruning**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model ../cache/llm_weights/models--facebook--opt-125m/snapshots/opt-125m \
    --sparsity_ratio 0.5 \
    --prune_method d2prune \
    --sparsity_type unstructured \
    --cali_dataset c4 \
    --cali_data_path ../cache/data/c4 \ # you should replace with your local directory
    --eval_dataset wikitext2 \
    --eval_data_path ../cache/data/wikitext \ # you should replace with your local directory
    --output_dir out/opt-125m-d2prune-sp0.5/ \
    --s 1500 \
    --r1 1 \
    --r2 0 \
    --d2_wanda \
    --d2_sparsegpt \
    --target_layer_names "['self_attn.k_proj']" \ # we provide the best optiml update configuration for q/k/v, no change is better!
    --eval_zero_shot
```

semi-structured pruing

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model ../cache/llm_weights/models--facebook--opt-125m/snapshots/opt-125m \
    --sparsity_ratio 0.5 \
    --prune_method d2prune \
    --sparsity_type 2:4 \
    --prune_n 2 \
    --prune_m 4 \
    --cali_dataset c4 \
    --cali_data_path ../cache/data/c4 \ # you should replace with your local directory
    --eval_dataset wikitext2 \
    --eval_data_path ../cache/data/wikitext \ # you should replace with your local directory
    --output_dir out/opt-125m-d2prune-n2m4/ \
    --s 1500 \
    --r1 1 \
    --r2 0 \
    --d2_wanda \
    --d2_sparsegpt \
    --target_layer_names "['self_attn.k_proj']" \ # we provide the best optiml update configuration for q/k/v, no change is better!
    --eval_zero_shot
```

We provide a quick overview of the arguments:

* `--model`: The identifier for the  model path on the local directory.
* `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
* `--prune_method`: We have implemented 4 pruning methods, namely [`sparsegpt`, `wanda`, `pruner-zero`, `d2pruneo`].
* `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`, `3:4`,].
* `--cali_dataset`: Calibration dataset name.
* `--cali_dataset_path`: Calibration dataset local directory.
* `--eval_dataset`: Evaluation dataset name.
* `--eval_dataset_path`: Evaluation dataset local directory.
* `--output_dir`: Specifies the directory where the result will be stored.
* `--s`: Scaling factor for input and output activations (it can be replaced "auto_s" to avoid parameter adjustment).
* `--r1`: .Perturbation coefficients of the first-order activation derivative term.
* `--r2`: .Perturbation coefficients of the second-order activation derivative term.
* `--d2_wanda`: Whether to use non-weight update pruning based on dual Taylor expansion for Q/K/V.
* `--d2_sparsegpt`: Whether to use weight update pruning based on dual Taylor expansion for MLP/FFN.
* `--target_layer_name`: The target layer in the attentional mechanism uses a non-weight update approach.
* `--eval_zero_shot`: Whether to evaluation for zero shot tasks.
* `--prune_n`: Parameter n for n:m pruning.
* `--prune_m`: Parameter m for n:m pruning.

**LoRA Finetuning  on C4**

```bash
CUDA_VISIBLE_DEVICES=0 python main_lora.py \
    --model_name_or_path "your_sparse_model_path" \
    --dataset_name c4 \
    --local_data_path ../cache/data/c4 \
    --data_cache_dir ./dataset/cache \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir "lora_ft_model_path" \
    --eval_zero_shot
```

## Acknowledgement

This repository is build upon the [Wanda](https://github.com/locuslab/wanda), [SparseGPT](https://github.com/IST-DASLab/sparsegpt), and [Pruner-Zero](https://github.com/pprp/Pruner-Zero) repositories. 

## License

This project is released under the MIT license. Please see the [LICENSE](https://github.com/pprp/Pruner-Zero/blob/main/LICENSE) file for more information.

## Citation

```bash
@article{Xiong_Liu_Ren_Bai_Fang_Zhang_Jiang_Tan_Liu_2026, 
    title={D2 Prune: Sparsifying Large Language Models via Dual Taylor Expansion and Attention Distribution Awareness}, 
    volume={40}, url={https://ojs.aaai.org/index.php/AAAI/article/view/39932}, DOI={10.1609/aaai.v40i32.39932},
    number={32}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Xiong, Lang and Liu, Ning and Ren, Ao and Bai, Yuheng and Fang, Haining and Zhang, Binyan and Jiang, Zhe and Tan, Yujuan and Liu, Duo}, 
    year={2026}, 
    month={Mar.}, 
    pages={27171-27179} 
}
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}

@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}

@inproceedings{dong2024pruner,
  title={Pruner-Zero: Evolving Symbolic Pruning Metric from Scratch for Large Language Models},
  author={Dong, Peijie and Li, Lujun and Tang, Zhenheng and Liu, Xiang and Pan, Xinglin and Wang, Qiang and Chu, Xiaowen},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024},
  organization={PMLR},
  url={https://arxiv.org/abs/2406.02924},
  note={[arXiv: 2406.02924]}
}
```
