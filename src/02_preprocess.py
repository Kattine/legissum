"""
02_preprocess.py
────────────────────────────────────────────────────────────
Filter raw BillSum data and convert to instruction fine-tuning format.
Run: python src/02_preprocess.py
"""

import os
import json
from tqdm import tqdm

RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS  = 14000   # ~3500 tokens; keeps GPU memory sane
MIN_TEXT_WORDS   = 200     # Drop trivially short bills
MIN_SUMMARY_WORDS = 20     # Drop near-empty summaries
MAX_SUMMARY_RATIO = 0.8    # Summary shouldn't be almost as long as text

SYSTEM_INSTRUCTION = (
    "You are a legislative assistant helping everyday citizens understand U.S. law. "
    "Read the following Congressional bill and write a clear, concise summary. "
    "Focus on: what the bill does, who it affects, and its key provisions. "
    "Use plain English — no legal jargon."
)


def is_quality(row: dict) -> bool:
    text    = row.get("text", "")
    summary = row.get("summary", "")
    if len(text.split())    < MIN_TEXT_WORDS:    return False
    if len(summary.split()) < MIN_SUMMARY_WORDS: return False
    if len(text)            > MAX_INPUT_CHARS * 2: return False
    if len(summary) / max(len(text), 1) > MAX_SUMMARY_RATIO: return False
    return True


def format_row(row: dict) -> dict:
    title   = row.get("title", "").strip()
    text    = row.get("text", "").strip()
    summary = row.get("summary", "").strip()

    # Combine title + text, truncate to token budget
    combined_input = f"{title}\n\n{text}"[:MAX_INPUT_CHARS]

    return {
        "instruction": SYSTEM_INSTRUCTION,
        "input":       combined_input,
        "output":      summary,
        # Alpaca-style single "text" field for TRL SFTTrainer
        "text": (
            f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
            f"### Bill:\n{combined_input}\n\n"
            f"### Summary:\n{summary}"
        ),
    }


def process_split(split_name: str):
    in_path  = os.path.join(RAW_DIR, f"{split_name}.jsonl")
    out_path = os.path.join(PROCESSED_DIR, f"{split_name}.jsonl")

    if not os.path.exists(in_path):
        print(f"  ⚠️  {in_path} not found — run 01_download_data.py first")
        return 0, 0

    rows, kept = [], []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    for row in tqdm(rows, desc=f"  {split_name}"):
        if is_quality(row):
            kept.append(format_row(row))

    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(rows), len(kept)


def main():
    print("🔧 Preprocessing BillSum splits...\n")
    for split in ["train", "test", "ca_test"]:
        total, kept = process_split(split)
        pct = (kept / total * 100) if total else 0
        print(f"  ✅ {split}: {total} → {kept} samples kept ({pct:.1f}%)")

    print(f"\n📁 Processed files saved to data/processed/")

    # Show one formatted example
    ex_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    if os.path.exists(ex_path):
        with open(ex_path) as f:
            ex = json.loads(f.readline())
        print("\n🔍 Formatted sample preview:")
        print(ex["text"][:600], "...\n")


if __name__ == "__main__":
    main()
