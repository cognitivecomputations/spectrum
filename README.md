# Spectrum

This repository contains the implementation of Spectrum, as detailed in the paper [Spectrum: Targeted Training on Signal to Noise Ratio](https://arxiv.org/abs/2406.06623).

## Overview

Spectrum is a tool for scanning and evaluating the Signal-to-Noise Ratio (SNR) of layers in large language models. By identifying the top n% of layers with the highest SNR, you can optimize training efficiency.

## Features

- **Model Scanning**: Scan models to determine the SNR of each layer.
- **Top n% Layer Identification**: Identify and sort the top n% of layers based on their SNR.
- **Unfrozen Parameters Configuration**: Generate configuration files for unfreezing specific layers in Axolotl or other libraries.

## Installation

To use Spectrum, you need to have Python installed. Clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/cognitivecomputations/spectrum.git
cd spectrum
pip install -r requirements.txt
```

## Usage

To use Spectrum, run the following command:

```bash
python spectrum.py --model-name <insert local or HF repo here> --top-percent <top % of snr ratios to target>
```

- `--model-name`: Specify the local model path or the Hugging Face repository.
- `--top-percent`: Specify the top percentage of SNR layers you want to retrieve.

Spectrum will check the `model_snr_results` folder to see if the model has already been scanned. If not, it will prompt you for the batch size to use for scanning. Once the scan is complete, it will output the SNR ratios in the `model_snr_results` folder and provide a sorted list of the highest to lowest SNR ratios along with an unfrozen parameters YAML file.

### Example Command

```bash
python spectrum.py --model-name meta-llama/Meta-Llama-3-8B-Instruct --top-percent 50
```

## Scanning New Models

If you want to scan a new model that is not in the `model_snr_results` folder, you can add your scans via a pull request (PR). Ensure you specify the batch size suitable for your hardware. For example, we have successfully used a batch size of 4 for 70B models on an 8xH100 node.

## Output Files

- **SNR Ratios**: Stored in the `model_snr_results` folder.
- **Unfrozen Parameters YAML**: A configuration file that matches an Axolotl config. You can copy and paste it directly into your Axolotl YAML file.

## Integration with Axolotl

If you're using Axolotl, the generated YAML file can be directly integrated.

## Integration with Other Libraries

For integration with other libraries, we provide a simple script to freeze and unfreeze parameters:

```python
def _freeze_and_unfreeze_parameters(self):
    # Freeze all parameters
    for param in self.model.parameters():
        param.requires_grad = False
    #unfreeze spectrum parameters
    for name, param in self.model.named_parameters():
        if any(unfrozen_param in name for unfrozen_param in self.unfrozen_parameters):
            param.requires_grad = True

unfrozen_parameters = [
    'model.layers.62.mlp.down_proj', 'model.layers.63.mlp.down_proj', 'model.layers.66.mlp.down_proj',
    'model.layers.65.mlp.down_proj', 'model.layers.64.mlp.down_proj', 'model.layers.67.mlp.down_proj',
    'model.layers.68.mlp.down_proj', 'model.layers.60.mlp.down_proj', 'model.layers.31.mlp.down_proj',
    'model.layers.69.mlp.down_proj', 'model.layers.61.mlp.down_proj', 'model.layers.59.mlp.down_proj',
    'model.layers.70.mlp.down_proj', 'model.layers.30.mlp.down_proj', 'model.layers.76.mlp.down_proj',
    'model.layers.72.mlp.down_proj', 'model.layers.77.mlp.down_proj', 'model.layers.71.mlp.down_proj',
    'model.layers.29.mlp.down_proj', 'model.layers.58.mlp.down_proj', 'model.layers.78.mlp.gate_proj',
    'model.layers.77.mlp.gate_proj', 'model.layers.76.mlp.gate_proj', 'model.layers.79.mlp.gate_proj',
    'model.layers.75.mlp.gate_proj', 'model.layers.74.mlp.gate_proj', 'model.layers.73.mlp.gate_proj',
    'model.layers.70.mlp.gate_proj', 'model.layers.72.mlp.gate_proj', 'model.layers.71.mlp.gate_proj',
    'model.layers.69.mlp.gate_proj', 'model.layers.54.mlp.gate_proj', 'model.layers.68.mlp.gate_proj',
    'model.layers.57.mlp.gate_proj', 'model.layers.63.mlp.gate_proj', 'model.layers.49.mlp.gate_proj',
    'model.layers.55.mlp.gate_proj', 'model.layers.53.mlp.gate_proj', 'model.layers.44.mlp.gate_proj',
    'model.layers.46.mlp.gate_proj', 'model.layers.69.mlp.up_proj', 'model.layers.70.mlp.up_proj',
    'model.layers.71.mlp.up_proj', 'model.layers.68.mlp.up_proj', 'model.layers.67.mlp.up_proj',
    'model.layers.66.mlp.up_proj', 'model.layers.46.mlp.up_proj', 'model.layers.63.mlp.up_proj',
    'model.layers.72.mlp.up_proj', 'model.layers.64.mlp.up_proj', 'model.layers.62.mlp.up_proj',
    'model.layers.45.mlp.up_proj', 'model.layers.65.mlp.up_proj', 'model.layers.73.mlp.up_proj',
    'model.layers.47.mlp.up_proj', 'model.layers.44.mlp.up_proj', 'model.layers.49.mlp.up_proj',
    'model.layers.48.mlp.up_proj', 'model.layers.53.mlp.up_proj', 'model.layers.74.mlp.up_proj',
    'model.layers.79.self_attn.k_proj', 'model.layers.36.self_attn.k_proj', 'model.layers.35.self_attn.k_proj',
    'model.layers.74.self_attn.k_proj', 'model.layers.34.self_attn.k_proj', 'model.layers.78.self_attn.k_proj',
    'model.layers.77.self_attn.k_proj', 'model.layers.37.self_attn.k_proj', 'model.layers.39.self_attn.k_proj',
    'model.layers.41.self_attn.k_proj', 'model.layers.38.self_attn.k_proj', 'model.layers.33.self_attn.k_proj',
    'model.layers.69.self_attn.k_proj', 'model.layers.42.self_attn.k_proj', 'model.layers.32.self_attn.k_proj',
    'model.layers.25.self_attn.k_proj', 'model.layers.70.self_attn.k_proj', 'model.layers.22.self_attn.k_proj',
    'model.layers.63.self_attn.k_proj', 'model.layers.29.self_attn.k_proj', 'model.layers.14.self_attn.o_proj',
    'model.layers.39.self_attn.o_proj', 'model.layers.19.self_attn.o_proj', 'model.layers.16.self_attn.o_proj',
    'model.layers.17.self_attn.o_proj', 'model.layers.15.self_attn.o_proj', 'model.layers.69.self_attn.o_proj',
    'model.layers.12.self_attn.o_proj', 'model.layers.42.self_attn.o_proj', 'model.layers.23.self_attn.o_proj',
    'model.layers.22.self_attn.o_proj', 'model.layers.29.self_attn.o_proj', 'model.layers.13.self_attn.o_proj',
    'model.layers.46.self_attn.o_proj', 'model.layers.52.self_attn.o_proj', 'model.layers.26.self_attn.o_proj',
    'model.layers.38.self_attn.o_proj', 'model.layers.41.self_attn.o_proj', 'model.layers.18.self_attn.o_proj',
    'model.layers.49.self_attn.o_proj', 'model.layers.1.self_attn.q_proj', 'model.layers.2.self_attn.q_proj',
    'model.layers.3.self_attn.q_proj', 'model.layers.5.self_attn.q_proj', 'model.layers.4.self_attn.q_proj',
    'model.layers.0.self_attn.q_proj', 'model.layers.6.self_attn.q_proj', 'model.layers.8.self_attn.q_proj',
    'model.layers.7.self_attn.q_proj', 'model.layers.9.self_attn.q_proj', 'model.layers.10.self_attn.q_proj',
    'model.layers.12.self_attn.q_proj', 'model.layers.19.self_attn.q_proj', 'model.layers.18.self_attn.q_proj',
    'model.layers.25.self_attn.q_proj', 'model.layers.11.self_attn.q_proj', 'model.layers.15.self_attn.q_proj',
    'model.layers.61.self_attn.q_proj', 'model.layers.17.self_attn.q_proj', 'model.layers.55.self_attn.q_proj',
    'model.layers.15.self_attn.v_proj', 'model.layers.16.self_attn.v_proj', 'model.layers.23.self_attn.v_proj',
    'model.layers.24.self_attn.v_proj', 'model.layers.25.self_attn.v_proj', 'model.layers.26.self_attn.v_proj',
    'model.layers.27.self_attn.v_proj', 'model.layers.28.self_attn.v_proj', 'model.layers.29.self_attn.v_proj',
    'model.layers.30.self_attn.v_proj', 'model.layers.31.self_attn.v_proj', 'model.layers.32.self_attn.v_proj',
    'model.layers.33.self_attn.v_proj', 'model.layers.34.self_attn.v_proj', 'model.layers.35.self_attn.v_proj',
    'model.layers.36.self_attn.v_proj', 'model.layers.37.self_attn.v_proj', 'model.layers.38.self_attn.v_proj',
    'model.layers.39.self_attn.v_proj', 'model.layers.41.self_attn.v_proj'
]
```

Replace `unfrozen_parameters` with the layers specific to your model.

## Example Configuration for Axolotl

```yaml
base_model: Qwen/Qwen2-72B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

trust_remote_code: true

# load_in_8bit: true
# load_in_4bit: false
# strict: false

datasets:
  - path: /workspace/datasets/dolphin-2.9.2/dolphin201-sharegpt2.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/dolphin-coder-codegen-sharegpt2.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/dolphin-coder-translate-sharegpt2.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/m-a-p_Code-Feedback-sharegpt-unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/m-a-p_CodeFeedback-Filtered-Instruction-sharegpt-unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/not_samantha_norefusals.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/openhermes200k_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/Orca-Math-resort-unfiltered.jsonl
    type: sharegpt  
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/SystemChat_sharegpt.jsonl
    type: sharegpt  
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/toolbench_instruct_j1s1_3k_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/toolbench_negative_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/toolbench_react_10p_unfiltered.jsonl
    type: sharegpt
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/toolbench_tflan_cot_30p_unfiltered.jsonl
    type: sharegpt 
    conversation: chatml
  - path: /workspace/datasets/dolphin-2.9.2/agent_instruct_react_unfiltered.jsonl
    type: sharegpt
    conversation: chatml

unfrozen_parameters:
- ^lm_head.weight$
- ^model.embed_tokens.weight$
# mlp.down_proj layers
- model.layers.62.mlp.down_proj
- model.layers.63.mlp.down_proj
- model.layers.66.mlp.down_proj
- model.layers.65.mlp.down_proj
- model.layers.64.mlp.down_proj
- model.layers.67.mlp.down_proj
- model.layers.68.mlp.down_proj
- model.layers.60.mlp.down_proj
- model.layers.31.mlp.down_proj
- model.layers.69.mlp.down_proj
- model.layers.61.mlp.down_proj
- model.layers.59.mlp.down_proj
- model.layers.70.mlp.down_proj
- model.layers.30.mlp.down_proj
- model.layers.76.mlp.down_proj
- model.layers.72.mlp.down_proj
- model.layers.77.mlp.down_proj
- model.layers.71.mlp.down_proj
- model.layers.29.mlp.down_proj
- model.layers.58.mlp.down_proj
- model.layers.75.mlp.down_proj
- model.layers.32.mlp.down_proj
- model.layers.56.mlp.down_proj
- model.layers.28.mlp.down_proj
- model.layers.26.mlp.down_proj
- model.layers.33.mlp.down_proj
- model.layers.34.mlp.down_proj
- model.layers.57.mlp.down_proj
- model.layers.27.mlp.down_proj
- model.layers.25.mlp.down_proj
- model.layers.35.mlp.down_proj
- model.layers.73.mlp.down_proj
- model.layers.24.mlp.down_proj
- model.layers.78.mlp.down_proj
- model.layers.74.mlp.down_proj
- model.layers.54.mlp.down_proj
# mlp.gate_proj layers
- model.layers.78.mlp.gate_proj
- model.layers.77.mlp.gate_proj
- model.layers.76.mlp.gate_proj
- model.layers.79.mlp.gate_proj
- model.layers.75.mlp.gate_proj
- model.layers.74.mlp.gate_proj
- model.layers.73.mlp.gate_proj
- model.layers.70.mlp.gate_proj
- model.layers.72.mlp.gate_proj
- model.layers.71.mlp.gate_proj
- model.layers.69.mlp.gate_proj
- model.layers.54.mlp.gate_proj
- model.layers.68.mlp.gate_proj
- model.layers.57.mlp.gate_proj
- model.layers.63.mlp.gate_proj
- model.layers.49.mlp.gate_proj
- model.layers.55.mlp.gate_proj
- model.layers.53.mlp.gate_proj
- model.layers.44.mlp.gate_proj
- model.layers.46.mlp.gate_proj
- model.layers.67.mlp.gate_proj
- model.layers.58.mlp.gate_proj
- model.layers.56.mlp.gate_proj
- model.layers.45.mlp.gate_proj
- model.layers.50.mlp.gate_proj
- model.layers.62.mlp.gate_proj
- model.layers.64.mlp.gate_proj
- model.layers.48.mlp.gate_proj
- model.layers.66.mlp.gate_proj
- model.layers.52.mlp.gate_proj
- model.layers.40.mlp.gate_proj
- model.layers.47.mlp.gate_proj
- model.layers.43.mlp.gate_proj
- model.layers.65.mlp.gate_proj
- model.layers.61.mlp.gate_proj
- model.layers.59.mlp.gate_proj
# mlp.up_proj layers
- model.layers.69.mlp.up_proj
- model.layers.70.mlp.up_proj
- model.layers.71.mlp.up_proj
- model.layers.68.mlp.up_proj
- model.layers.67.mlp.up_proj
- model.layers.66.mlp.up_proj
- model.layers.46.mlp.up_proj
- model.layers.63.mlp.up_proj
- model.layers.72.mlp.up_proj
- model.layers.64.mlp.up_proj
- model.layers.62.mlp.up_proj
- model.layers.45.mlp.up_proj
- model.layers.65.mlp.up_proj
- model.layers.73.mlp.up_proj
- model.layers.47.mlp.up_proj
- model.layers.44.mlp.up_proj
- model.layers.49.mlp.up_proj
- model.layers.48.mlp.up_proj
- model.layers.53.mlp.up_proj
- model.layers.74.mlp.up_proj
- model.layers.75.mlp.up_proj
- model.layers.57.mlp.up_proj
- model.layers.76.mlp.up_proj
- model.layers.43.mlp.up_proj
- model.layers.42.mlp.up_proj
- model.layers.61.mlp.up_proj
- model.layers.40.mlp.up_proj
- model.layers.56.mlp.up_proj
- model.layers.60.mlp.up_proj
- model.layers.31.mlp.up_proj
- model.layers.54.mlp.up_proj
- model.layers.55.mlp.up_proj
- model.layers.32.mlp.up_proj
- model.layers.41.mlp.up_proj
- model.layers.33.mlp.up_proj
- model.layers.58.mlp.up_proj
# self_attn.k_proj layers
- model.layers.79.self_attn.k_proj
- model.layers.36.self_attn.k_proj
- model.layers.35.self_attn.k_proj
- model.layers.74.self_attn.k_proj
- model.layers.34.self_attn.k_proj
- model.layers.78.self_attn.k_proj
- model.layers.77.self_attn.k_proj
- model.layers.37.self_attn.k_proj
- model.layers.39.self_attn.k_proj
- model.layers.41.self_attn.k_proj
- model.layers.38.self_attn.k_proj
- model.layers.33.self_attn.k_proj
- model.layers.69.self_attn.k_proj
- model.layers.42.self_attn.k_proj
- model.layers.32.self_attn.k_proj
- model.layers.25.self_attn.k_proj
- model.layers.70.self_attn.k_proj
- model.layers.22.self_attn.k_proj
- model.layers.63.self_attn.k_proj
- model.layers.29.self_attn.k_proj
- model.layers.68.self_attn.k_proj
- model.layers.24.self_attn.k_proj
- model.layers.30.self_attn.k_proj
- model.layers.66.self_attn.k_proj
- model.layers.31.self_attn.k_proj
- model.layers.23.self_attn.k_proj
- model.layers.65.self_attn.k_proj
- model.layers.57.self_attn.k_proj
- model.layers.28.self_attn.k_proj
- model.layers.64.self_attn.k_proj
- model.layers.44.self_attn.k_proj
- model.layers.27.self_attn.k_proj
- model.layers.75.self_attn.k_proj
- model.layers.40.self_attn.k_proj
- model.layers.26.self_attn.k_proj
- model.layers.61.self_attn.k_proj
# self_attn.o_proj layers
- model.layers.14.self_attn.o_proj
- model.layers.39.self_attn.o_proj
- model.layers.19.self_attn.o_proj
- model.layers.16.self_attn.o_proj
- model.layers.17.self_attn.o_proj
- model.layers.15.self_attn.o_proj
- model.layers.69.self_attn.o_proj
- model.layers.12.self_attn.o_proj
- model.layers.42.self_attn.o_proj
- model.layers.23.self_attn.o_proj
- model.layers.22.self_attn.o_proj
- model.layers.29.self_attn.o_proj
- model.layers.13.self_attn.o_proj
- model.layers.46.self_attn.o_proj
- model.layers.52.self_attn.o_proj
- model.layers.26.self_attn.o_proj
- model.layers.38.self_attn.o_proj
- model.layers.41.self_attn.o_proj
- model.layers.18.self_attn.o_proj
- model.layers.49.self_attn.o_proj
- model.layers.11.self_attn.o_proj
- model.layers.28.self_attn.o_proj
- model.layers.25.self_attn.o_proj
- model.layers.47.self_attn.o_proj
- model.layers.53.self_attn.o_proj
- model.layers.27.self_attn.o_proj
- model.layers.37.self_attn.o_proj
- model.layers.20.self_attn.o_proj
- model.layers.43.self_attn.o_proj
- model.layers.44.self_attn.o_proj
- model.layers.45.self_attn.o_proj
- model.layers.30.self_attn.o_proj
- model.layers.24.self_attn.o_proj
- model.layers.21.self_attn.o_proj
- model.layers.10.self_attn.o_proj
- model.layers.3.self_attn.o_proj
# self_attn.q_proj layers
- model.layers.1.self_attn.q_proj
- model.layers.2.self_attn.q_proj
- model.layers.3.self_attn.q_proj
- model.layers.5.self_attn.q_proj
- model.layers.4.self_attn.q_proj
- model.layers.0.self_attn.q_proj
- model.layers.6.self_attn.q_proj
- model.layers.8.self_attn.q_proj
- model.layers.7.self_attn.q_proj
- model.layers.9.self_attn.q_proj
- model.layers.10.self_attn.q_proj
- model.layers.12.self_attn.q_proj
- model.layers.19.self_attn.q_proj
- model.layers.18.self_attn.q_proj
- model.layers.25.self_attn.q_proj
- model.layers.11.self_attn.q_proj
- model.layers.15.self_attn.q_proj
- model.layers.61.self_attn.q_proj
- model.layers.17.self_attn.q_proj
- model.layers.55.self_attn.q_proj
- model.layers.54.self_attn.q_proj
- model.layers.16.self_attn.q_proj
- model.layers.68.self_attn.q_proj
- model.layers.49.self_attn.q_proj
- model.layers.48.self_attn.q_proj
- model.layers.52.self_attn.q_proj
- model.layers.13.self_attn.q_proj
- model.layers.42.self_attn.q_proj
- model.layers.57.self_attn.q_proj
- model.layers.60.self_attn.q_proj
- model.layers.53.self_attn.q_proj
- model.layers.64.self_attn.q_proj
- model.layers.66.self_attn.q_proj
- model.layers.62.self_attn.q_proj
- model.layers.59.self_attn.q_proj
- model.layers.50.self_attn.q_proj
# self_attn.v_proj layers
- model.layers.15.self_attn.v_proj
- model.layers.16.self_attn.v_proj
- model.layers.23.self_attn.v_proj
- model.layers.24.self_attn.v_proj
- model.layers.25.self_attn.v_proj
- model.layers.26.self_attn.v_proj
- model.layers.27.self_attn.v_proj
- model.layers.28.self_attn.v_proj
- model.layers.29.self_attn.v_proj
- model.layers.30.self_attn.v_proj
- model.layers.31.self_attn.v_proj
- model.layers.32.self_attn.v_proj
- model.layers.33.self_attn.v_proj
- model.layers.34.self_attn.v_proj
- model.layers.35.self_attn.v_proj
- model.layers.36.self_attn.v_proj
- model.layers.37.self_attn.v_proj
- model.layers.38.self_attn.v_proj
- model.layers.39.self_attn.v_proj
- model.layers.41.self_attn.v_proj
- model.layers.42.self_attn.v_proj
- model.layers.48.self_attn.v_proj
- model.layers.53.self_attn.v_proj
- model.layers.57.self_attn.v_proj
- model.layers.58.self_attn.v_proj
- model.layers.59.self_attn.v_proj
- model.layers.61.self_attn.v_proj
- model.layers.63.self_attn.v_proj
- model.layers.64.self_attn.v_proj
- model.layers.65.self_attn.v_proj
- model.layers.66.self_attn.v_proj
- model.layers.69.self_attn.v_proj
- model.layers.74.self_attn.v_proj
- model.layers.75.self_attn.v_proj
- model.layers.76.self_attn.v_proj
- model.layers.72.self_attn.v_proj

  
chat_template: chatml
dataset_prepared_path: qwen2-72b-data
val_set_size: 0.01
output_dir: qwen2-72b

sequence_len: 8192  # supports up to 8192
sample_packing: true
pad_to_sequence_len: true

# adapter: lora
# lora_model_dir:
# lora_r: 32
# lora_alpha: 16
# lora_dropout: 0.05
# lora_target_linear: true
# lora_fan_in_fan_out:

wandb_project: qwen2-72b
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 3
optimizer: paged_adamw_8bit
lr_scheduler: cosine
learning_rate: 1e-5

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 2
eval_table_size:
eval_max_new_tokens: 128
saves_per_epoch: 4
save_total_limit: 2
debug:
deepspeed: /workspace/axolotl/deepspeed_configs/zero3_bf16_cpuoffload_params.json
weight_decay: 0.05
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|endoftext|>"
  eos_token: "<|im_end|>"

```

## Contribution

We invite you to contribute to the Spectrum project by adding new model scans. Please fork the repository, add your scans to the `model_snr_results` folder, and submit a pull request.

## License

This project is licensed under the Apache-2.0 License.

## Acknowledgments

Thank you for using Spectrum! For any questions or issues, please open an issue on this repository.
```
