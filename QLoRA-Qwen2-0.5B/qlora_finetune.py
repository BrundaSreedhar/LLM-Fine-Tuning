#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script: Clinical JSON Extraction with Qwen2-0.5B

Usage:
    pip install transformers accelerate bitsandbytes peft "datasets<3.0.0" sentencepiece
    python qlora_finetune.py
"""

import argparse
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-0.5B"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "o_proj"]
LORA_DROPOUT = 0.05
MAX_SEQ_LEN = 256
TRAIN_SAMPLES = 150
VAL_SAMPLES = 30
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
OUTPUT_DIR = "./qlora-qwen-output"
ADAPTER_DIR = "./qlora-qwen-adapter"
MERGED_DIR = "./qlora-qwen-merged"
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_disease_entities(tokens, ner_tags):
    """Extract disease spans from BIO tags (1=B-Disease, 2=I-Disease)."""
    diseases, current = [], []
    for token, tag in zip(tokens, ner_tags):
        if tag == 1:
            if current:
                diseases.append(" ".join(current))
            current = [token]
        elif tag == 2:
            current.append(token)
        else:
            if current:
                diseases.append(" ".join(current))
                current = []
    if current:
        diseases.append(" ".join(current))
    return diseases


def build_prompt(input_text, json_output=None):
    """Build the prompt string. If json_output is provided, append it (training).
    Otherwise return the prompt up to '### OUTPUT:' (inference)."""
    base = (
        '### SYSTEM: You are a clinical data parser. Extract the core entities '
        'from the medical text below. You must output ONLY a valid JSON object '
        'with the keys "subject", "disease", and "outcome".\n'
        f'### INPUT: {input_text}\n'
        '### OUTPUT:'
    )
    if json_output is not None:
        return base + " " + json_output
    return base


def make_format_fn(tokenizer):
    """Return a map-compatible formatting function (closure over tokenizer)."""

    def format_example(example):
        input_text = " ".join(example["tokens"])
        diseases = extract_disease_entities(example["tokens"], example["ner_tags"])
        disease_str = ", ".join(diseases) if diseases else "none"

        json_output = json.dumps({
            "subject": input_text[:60],
            "disease": disease_str,
            "outcome": "mentioned in clinical abstract",
        })

        full_prompt = build_prompt(input_text, json_output)
        prompt_only = build_prompt(input_text)  # without answer

        tokenized = tokenizer(full_prompt, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        labels = tokenized["input_ids"].copy()

        # Mask prompt portion so loss is only on the JSON output
        prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return tokenized

    return format_example


def extract_first_json(text):
    """Extract the first complete JSON object from a string."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for clinical JSON extraction")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=VAL_SAMPLES)
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only run inference from saved adapter")
    args = parser.parse_args()

    # ---- 1. Quantization config ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ---- 2. Tokenizer ----
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.skip_training:
        # ---- 3. Load model in 4-bit ----
        print(f"Loading {MODEL_ID} in 4-bit NF4 with double quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        # ---- 4. Attach LoRA adapter ----
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ---- 5. Prepare dataset ----
        print("Loading ncbi_disease dataset...")
        raw_dataset = load_dataset("ncbi_disease", trust_remote_code=True)
        format_fn = make_format_fn(tokenizer)

        train_data = raw_dataset["train"].shuffle(seed=SEED).select(range(args.train_samples))
        val_data = raw_dataset["validation"].shuffle(seed=SEED).select(range(args.val_samples))

        train_tokenized = train_data.map(format_fn, remove_columns=["id", "tokens", "ner_tags"])
        val_tokenized = val_data.map(format_fn, remove_columns=["id", "tokens", "ner_tags"])
        print(f"Train: {len(train_tokenized)}  |  Val: {len(val_tokenized)}")

        # ---- 6. Train ----
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=10,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            report_to="none",
            learning_rate=args.lr,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        print("Starting QLoRA fine-tuning...")
        trainer.train()
        print("Training complete.")

        # ---- 7. Save adapter ----
        print(f"Saving LoRA adapter to {ADAPTER_DIR}...")
        model.save_pretrained(ADAPTER_DIR)

    # ---- 8. Merge adapter into full-precision model ----
    print("Reloading base model in fp16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    merged_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = merged_model.merge_and_unload()
    merged_model.eval()
    print("Adapter merged successfully.")

    # ---- 9. Inference test ----
    test_passages = [
        "Somatic mutation of the ATM gene contributes to the development of T-PLL tumors.",
        (
            "A retrospective study of 95 pediatric patients with Duchenne muscular dystrophy "
            "found that early corticosteroid intervention preserved ambulation in 70% of cases "
            "at the 5-year assessment."
        ),
    ]

    device = next(merged_model.parameters()).device
    for passage in test_passages:
        prompt = build_prompt(passage)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print(f"\n{'='*60}")
        print(f"Passage: {passage[:80]}...")

        with torch.no_grad():
            out_ids = merged_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Raw output: {raw_output}")

        json_str = extract_first_json(raw_output)
        try:
            parsed = json.loads(json_str)
            print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
        except (json.JSONDecodeError, TypeError):
            print("Failed to parse valid JSON from model output.")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
