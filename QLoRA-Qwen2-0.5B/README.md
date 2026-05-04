# QLoRA Clinical JSON Extraction

Fine-tune [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) with QLoRA to extract structured JSON from clinical text. The model learns to parse biomedical abstracts from the [ncbi_disease](https://huggingface.co/datasets/ncbi_disease) dataset and output a JSON object with `subject`, `disease`, and `outcome` fields.

This project is an example of fine-tuning a small language model to follow a strict output structure. Rather than relying on prompt engineering alone, it uses QLoRA to teach Qwen2-0.5B to consistently produce valid JSON with a fixed schema (subject, disease, outcome) when given free-form text. The clinical NER task from the ncbi_disease dataset is just the use case — the same approach generalizes to any scenario where you need a model to reliably output structured data in a specific format.

## Overview

This project demonstrates parameter-efficient fine-tuning under a strict memory budget. The LoRA adapter adds **~0.30–0.60%** trainable parameters on top of a 4-bit quantized base model, making it feasible to train on a single consumer GPU or a free Colab instance.

### Pipeline at a glance

1. **Load** the base model in 4-bit NormalFloat (NF4) with double quantization via `bitsandbytes`.
2. **Attach** a LoRA adapter targeting `q_proj`, `v_proj`, and `o_proj` (rank 16, alpha 32).
3. **Prepare data** from the `ncbi_disease` dataset — each example is formatted into a prompt/completion pair where the completion is a JSON object. Loss is masked so the model only learns to generate the JSON output.
4. **Train** for 1–3 epochs with the Hugging Face `Trainer`.
5. **Merge** the adapter back into a full-precision model and run inference on unseen clinical passages.

## Requirements

- Python 3.9+
- A CUDA-capable GPU (tested on Colab T4 / A100)
- ~4 GB VRAM during training

## Setup

```bash
# Clone the repo
git clone https://github.com/BrundaSreedhar/QLoRA-Qwen2-0.5B.git
cd QLoRA-Qwen2-0.5B

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic run (train + inference)

```bash
python qlora_finetune.py
```

This will train for 3 epochs on 150 samples, save the adapter, merge it, and run two test passages.

### Custom settings

```bash
python qlora_finetune.py \
  --epochs 1 \
  --batch-size 8 \
  --lr 3e-4 \
  --train-samples 300 \
  --val-samples 50
```

### Skip training (inference only from a saved adapter)

```bash
python qlora_finetune.py --skip-training
```

This assumes the adapter has already been saved to `./qlora-qwen-adapter`.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-device train and eval batch size |
| `--lr` | `2e-4` | Learning rate |
| `--train-samples` | `150` | Number of training examples to use |
| `--val-samples` | `30` | Number of validation examples to use |
| `--skip-training` | off | Skip training and jump to merge + inference |

## Example output

Given the input:

> Somatic mutation of the ATM gene contributes to the development of T-PLL tumors.

The fine-tuned model produces:

```json
{
  "subject": "Somatic mutation of the ATM gene contributes to the develo",
  "disease": "T-PLL tumors",
  "outcome": "mentioned in clinical abstract"
}
```

## Project structure

```
.
├── qlora_finetune.py        # Main training and inference script
├── README.md
├── qlora-qwen-output/       # Trainer checkpoints (created at runtime)
└── qlora-qwen-adapter/      # Saved LoRA adapter weights (created at runtime)
```

## How it works

### Quantization

The base model is loaded in 4-bit NF4 precision with double quantization enabled (`BitsAndBytesConfig`). This cuts memory usage roughly 4× compared to fp16 while preserving model quality for fine-tuning.

### LoRA configuration

| Parameter | Value | Rationale |
|---|---|---|
| `r` | 16 | Rank of the low-rank matrices |
| `lora_alpha` | 32 | 2:1 alpha-to-rank ratio for stable learning rate scaling |
| `target_modules` | `q_proj, v_proj, o_proj` | Three attention projections hit the 0.30–0.60% trainable parameter budget |
| `dropout` | 0.05 | Light regularization |

### Data formatting

Each `ncbi_disease` example is converted into a prompt like:

```
### SYSTEM: You are a clinical data parser. Extract the core entities ...
### INPUT: <reconstructed abstract text>
### OUTPUT: {"subject": "...", "disease": "...", "outcome": "..."}
```

Labels are masked with `-100` on the prompt portion so the cross-entropy loss is computed only over the JSON output tokens.
