<p align="center" width="100%">
<img src="assets/images/logo.gif" alt="Stable Alignment" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</p>

# Stable Alignment - Alignment Learning in Social Games

[![lint](https://github.com/DapangLiu/SandBox/actions/workflows/code_quality.yml/badge.svg)](https://github.com/DapangLiu/SandBox/blob/main/.github/workflows/code_quality.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official repo for the Stable Alignment project. We aim to provide a RLHF alternative which is superior in alignment performance, highly-efficient in data learning, and easy to deploy in scaled-up settings. Instead of training an extra reward model that can be gamed during optimization, we directly train on the recorded interaction data in simulated social games. We find high-quality data + reliable algorithm is the secret recipe for stable alignment learning.

The repo contains:

- The code for [running social simulation in Sandbox](#sandbox-simulation).
- The [169K interaction data](#data-release) used for alignment training.
- The code for [training with stable alignment](#training-with-stable-alignment).
- The download for [So(cially)-Good Language Model](#downloading-model).

**Life is a game. Play by your rules!**

<p>
<img src="assets/images/overview.png" alt="Overview of Stable Alignment" style="width: 100%; min-width: 200px; display: block; margin: auto;">
</p>

## Sandbox Simulation

### Installation
```bash
# install development environment
pip install -r requirements.txt
# install dependencies for package re-distribution
pip install -e .
```
### Simulation Setup
- Initial data is already stored at `assets/hh-rlhf/labeled_prior.jsonl` (with Git LFS). 
- After a round of simulation, the simulated interaction data and metrics will be saved at `data/cache/world_<world_id>/`.
- Place your OpenAI API key in `.env` inside the project root folder.

### Run Simulation
Navigate to the project root folder and run simulation with customized settings:

```bash
python stable_alignment/simulation.py \
    -model_type 'text-davinci-002' \
    -obs_model_type 'gpt-3.5-turbo' \
    -world_id 1 \
    -init_setting 'all_bad' \
    -n_round '2' \
    -size '4' \
    -dataset_name 'hh-rlhf'
```

## Alignment Data Release

<p>
<img src="assets/images/back_scatter.png" alt="Back Scatter in SandBox" style="width: 100%; min-width: 200px; display: block; margin: auto;">
</p>

The alignment data used for training has been already included in the path `assets/sandbox_v1.json` and `assets/sandbox_v2.json`. Note that they are sampled from the full set of interaction data by a ratio of 5:1:1 for Alignment Imitation, Self-Critic, and Realignment data respectively. The full set of interaction data is available upon request.

<details>
<summary> <strong> The Statistics of Alignment Data (Full Set) </strong> </summary>

- `sandbox_v1.json`

| Data / Social Agent Type | text-davinci-002 | text-davinci-003 | ChatGPT | Total |
|--------------------------|------------------|------------------|---------|-------|
| Alignment Imitation      | 9.8k             | 10k              | 10k     | 29.8k |
| Self-Critic              | 17k              | 20k              | 20k     | 57k   |
| Realignment              | 3.3k             | 3k               | 0.7k    | 7k    |
| Total                    | 30.1k            | 33k              | 30.7k   | 93.8k |

- `sandbox_v2.json`

| Data / Social Agent Type | text-davinci-002 | text-davinci-003 | GPT4  | Total |
|--------------------------|------------------|------------------|-------|-------|
| Alignment Imitation      | 18.2k            | 10.4k            | 20.2k | 48.8k |
| Self-Critic              | 36.3k            | 18.3k            | 40k   | 94.6k |
| Realignment              | 18.2k            | 3.4k             | 4.0k  | 25.6k |
| Total                    | 72.7k            | 32.1k            | 64.2k | 169k  |

</details>

## Training with Stable Alignment

```bash
torchrun --nproc_per_node=4 --master_port=36646 train_alignment.py \
      --model_name_or_path "/workspace/hhh_sft" \  # path to your SFT model
      --data_path "./assets/sandbox_v1.json" \ # path to the alignment data
      --bf16 True \
      --output_dir "/workspace/<your_output_lm_name>" \
      --num_train_epochs 7 \
      --per_device_train_batch_size 1 \  # batch size has to be 1 for alignment training
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 200 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "shard_grad_op auto_wrap" \  # change to "full_shard auto_wrap" if OOM
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True \
      --model_max_length 360 \  # change to shorter length if OOM
      --rating_scale 7 \  # the scale of the ratings. 7 for 1-7, 10 for 1-10, etc.
      --margin 10 \  # constant, see the paper
      --max_flow False \  # mean or max for the penalty
      --ratio 0.2 \  # control the ratio of the penalty
      --num_comp 3
```

## So(cially)-Good Language Model

![Model Release](assets/images/model_select_light.png#gh-light-mode-only)
![Model Release](assets/images/model_select_dark.png#gh-dark-mode-only)

We have released our models on huggingface! 🤗

Released models include:

1. [`better-base`](https://huggingface.co/agi-css/better-base), base model trained on LLaMA with [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) which is the fixed Alpaca instruction tuning dataset, and [codealpaca](https://github.com/sahil280114/codealpaca) which is the code pretraining dataset.

2. [`hh-rlhf-sft`](https://huggingface.co/agi-css/hh-rlhf-sft), supervised fine-tuned model on `better-base` with the socially aligned demonstrations in [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) (the `accepted` samples in the dataset).
3. [`socially-good-lm`](https://huggingface.co/agi-css/socially-good-lm), socially aligned language model trained on `hh-rlhf-sft` with the stable alignment method.

After you download the model, you can run inference with the following command:

```bash
python stable_alignment/run_inference.py \
    --model_path './models/socially-good-lm' \
    --device 'cuda:0'
```

# Citation

Please cite our paper if you use the data or code in this repo:

```bibtex
@misc{liu2023sociallyaligned,
      title={Training Socially Aligned Language Models in Simulated Human Society},
      author={Ruibo Liu and Ruixin Yang and Chenyan Jia and Ge Zhang and Denny Zhou and Andrew M. Dai and Diyi Yang and Soroush Vosoughi},
      year={2023},
      eprint={2305.16960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
