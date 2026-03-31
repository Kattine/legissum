"""
04_evaluate.py
────────────────────────────────────────────────────────────
Before/After comparison + ROUGE evaluation.
Run: python src/04_evaluate.py
"""

import os
import json
import torch
import csv
from tqdm import tqdm
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "billsum-lora")
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

BASE_MODEL    = "mistralai/Mistral-7B-Instruct-v0.3"
N_EVAL        = 50          # How many test samples to score
N_DISPLAY     = 3           # How many to print in full for demo

SYSTEM_INSTRUCTION = (
    "You are a legislative assistant helping everyday citizens understand U.S. law. "
    "Read the following Congressional bill and write a clear, concise summary. "
    "Focus on: what the bill does, who it affects, and its key provisions. "
    "Use plain English — no legal jargon."
)


def load_model(base_model_id: str, adapter_path: str | None = None):
    # Use MPS on Apple Silicon, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  ✅ Loaded LoRA adapters from {adapter_path}")
    else:
        print("  ⚠️  No adapter found — running base model only")

    return model, tokenizer


def summarize(model, tokenizer, bill_text: str, max_new_tokens: int = 300) -> str:
    prompt = (
        f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Bill:\n{bill_text[:3000]}\n\n"
        f"### Summary:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract only the summary part
    if "### Summary:" in decoded:
        return decoded.split("### Summary:")[-1].strip()
    return decoded.strip()


def score_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        for k in agg:
            agg[k] += s[k].fmeasure
    n = len(predictions)
    return {k: round(v / n, 4) for k, v in agg.items()}


def main():
    # ── Load test data ─────────────────────────────────────────────────────────
    test_path = os.path.join(PROCESSED_DIR, "test.jsonl")
    assert os.path.exists(test_path), "Run 02_preprocess.py first!"
    with open(test_path) as f:
        test_data = [json.loads(l) for l in f][:N_EVAL]
    print(f"📋 Evaluating on {len(test_data)} test samples\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("⬇️  Loading base model (BEFORE)...")
    base_model, tokenizer = load_model(BASE_MODEL, adapter_path=None)

    print("\n⬇️  Loading fine-tuned model (AFTER)...")
    ft_model, _ = load_model(BASE_MODEL, adapter_path=MODEL_OUT_DIR)

    # ── Generate & compare ────────────────────────────────────────────────────
    results = []
    preds_base, preds_ft, refs = [], [], []

    print("\n🔄 Generating summaries...\n")
    for i, example in enumerate(tqdm(test_data)):
        ref        = example["output"]
        pred_base  = summarize(base_model, tokenizer, example["input"])
        pred_ft    = summarize(ft_model,   tokenizer, example["input"])

        preds_base.append(pred_base)
        preds_ft.append(pred_ft)
        refs.append(ref)

        results.append({
            "title":      example["input"][:80],
            "reference":  ref,
            "base_model": pred_base,
            "finetuned":  pred_ft,
        })

        # Live Before/After display for first N_DISPLAY samples
        if i < N_DISPLAY:
            print("=" * 70)
            print(f"📄 Bill: {example['input'][:200]}...\n")
            print(f"❌ BEFORE (base):\n{pred_base}\n")
            print(f"✅ AFTER  (fine-tuned):\n{pred_ft}\n")
            print(f"🏛️  Ground truth:\n{ref}\n")

    # ── ROUGE scores ──────────────────────────────────────────────────────────
    scores_base = score_rouge(preds_base, refs)
    scores_ft   = score_rouge(preds_ft,   refs)

    print("\n📊 ROUGE Scores")
    print(f"{'Metric':<12} {'Base Model':>12} {'Fine-tuned':>12} {'Δ Improvement':>14}")
    print("-" * 52)
    for k in ["rouge1", "rouge2", "rougeL"]:
        delta = scores_ft[k] - scores_base[k]
        print(f"{k:<12} {scores_base[k]:>12.4f} {scores_ft[k]:>12.4f} {delta:>+14.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_csv = os.path.join(OUTPUTS_DIR, "eval_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    out_scores = os.path.join(OUTPUTS_DIR, "rouge_scores.json")
    with open(out_scores, "w") as f:
        json.dump({"base": scores_base, "finetuned": scores_ft}, f, indent=2)

    # Save a few rich examples for the Gradio demo (Tab 2 — Before vs After)
    sample_comparisons = results[:5]
    out_samples = os.path.join(OUTPUTS_DIR, "sample_comparisons.json")
    with open(out_samples, "w", encoding="utf-8") as f:
        json.dump(sample_comparisons, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to outputs/")


if __name__ == "__main__":
    main()
