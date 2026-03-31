"""
train_local_mps.py — Fine-tune Mistral-7B on Apple Silicon (M1/M2/M3/M4)
────────────────────────────────────────────────────────────────────────────
No CUDA, no bitsandbytes. Uses MPS backend + LoRA in float16.

Run:
    python src/train_local_mps.py

Requirements (already in requirements.txt):
    pip install transformers peft trl datasets accelerate

HF login required (Mistral is a gated model):
    huggingface-cli login
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_OUT   = str(Path(__file__).parent.parent / "models" / "billsum-lora")
DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"

MAX_TRAIN   = 500   # ~30-60 min on M4 Pro; raise to 1500 for better quality
MAX_SEQ_LEN = 1024
EPOCHS      = 1

HF_USERNAME  = os.environ.get("HF_USERNAME", "")
HUB_MODEL_ID = f"{HF_USERNAME}/legissum-mistral-lora" if HF_USERNAME else None

SYSTEM = (
    "You are a legislative assistant helping everyday citizens understand U.S. law. "
    "Read the following Congressional bill and write a clear, concise summary. "
    "Focus on: what the bill does, who it affects, and its key provisions. "
    "Use plain English -- no legal jargon."
)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = "cpu"
    print("MPS not available, falling back to CPU (will be slow)")

# ── Data ──────────────────────────────────────────────────────────────────────
train_jsonl = DATA_DIR / "train.jsonl"

if train_jsonl.exists():
    print(f"Loading existing data from {train_jsonl}")
    with open(train_jsonl) as f:
        rows = [json.loads(l) for l in f][:MAX_TRAIN]
else:
    print("Downloading BillSum dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset("FiscalNote/billsum")
    NL = chr(10)
    rows = []
    for row in ds["train"]:
        if len(rows) >= MAX_TRAIN:
            break
        if len(row["text"].split()) < 200 or len(row["summary"].split()) < 20:
            continue
        title = row["title"]
        text  = row["text"]
        summ  = row["summary"]
        inp   = (title + NL + NL + text)[:14000]
        rows.append({
            "input":  inp,
            "output": summ,
            "text": ("### Instruction:" + NL + SYSTEM
                     + NL + NL + "### Bill:" + NL + inp
                     + NL + NL + "### Summary:" + NL + summ),
        })
    with open(train_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + NL)

train_ds = Dataset.from_list(rows)
print(f"Training on {len(train_ds)} samples")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Model (float16, no quantization) ─────────────────────────────────────────
print(f"Loading {BASE_MODEL} in float16...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.to(DEVICE)
model.enable_input_require_grads()

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=8,                      # smaller r = less memory, still effective
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],   # fewer modules = faster
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_OUT, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir=MODEL_OUT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        dataset_text_field="text",
        report_to="none",
        warmup_steps=20,
        push_to_hub=bool(HUB_MODEL_ID),
        hub_model_id=HUB_MODEL_ID,
    ),
)

print("Training started...")
trainer.train()
trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
print(f"Adapter saved to {MODEL_OUT}")

if HUB_MODEL_ID:
    print(f"Pushing to HF Hub: {HUB_MODEL_ID}")
    model.push_to_hub(HUB_MODEL_ID)
    tokenizer.push_to_hub(HUB_MODEL_ID)
    print(f"Done. Set HF_MODEL_ID={HUB_MODEL_ID} in your .env")
