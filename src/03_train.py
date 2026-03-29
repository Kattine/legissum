"""
03_train.py
────────────────────────────────────────────────────────────
QLoRA fine-tuning of Mistral-7B-Instruct on BillSum.
Run: python src/03_train.py
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "billsum-lora")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"
# Lighter alternative if VRAM is tight:
# BASE_MODEL    = "meta-llama/Llama-3.2-3B-Instruct"

LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "v_proj", "k_proj", "o_proj"]

EPOCHS          = 2
BATCH_SIZE      = 2
GRAD_ACCUM      = 4     # effective batch = 8
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 1024
LOGGING_STEPS   = 50
SAVE_STEPS      = 500


def load_jsonl(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    return Dataset.from_list(data)


def main():
    print(f"🤖 Base model : {BASE_MODEL}")
    print(f"💾 Output dir : {MODEL_OUT_DIR}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    assert os.path.exists(train_path), "Run 02_preprocess.py first!"
    train_ds = load_jsonl(train_path)
    print(f"📊 Training on {len(train_ds)} samples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit quantization config (QLoRA) ─────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load base model ───────────────────────────────────────────────────────
    print("⬇️  Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=MODEL_OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        report_to="none",               # set "wandb" if you want tracking
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=training_args,
        tokenizer=tokenizer,
    )

    print("\n🚀 Starting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    print(f"\n✅ Model saved to {MODEL_OUT_DIR}")
    print("   (Upload adapters to HuggingFace Hub with: "
          "model.push_to_hub('your-username/legissum-lora'))")


if __name__ == "__main__":
    main()
