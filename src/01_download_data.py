"""
01_download_data.py
────────────────────────────────────────────────────────────
Download BillSum from HuggingFace and save raw splits locally.
Run: python src/01_download_data.py
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def main():
    print("📥 Downloading BillSum from HuggingFace (~67MB)...")
    ds = load_dataset("FiscalNote/billsum")

    for split_name, split_data in ds.items():
        out_path = os.path.join(RAW_DIR, f"{split_name}.jsonl")
        print(f"  Saving {split_name} ({len(split_data)} rows) → {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in tqdm(split_data, desc=split_name):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n✅ Done. Files saved to data/raw/")
    print("\n📊 Quick stats:")
    for split_name, split_data in ds.items():
        texts   = [len(r["text"].split())    for r in split_data]
        sums    = [len(r["summary"].split()) for r in split_data]
        avg_compression = (sum(texts) / len(texts)) / (sum(sums) / len(sums))
        print(f"  {split_name:10s}: {len(split_data):5d} samples | "
              f"avg text {sum(texts)//len(texts):5d} words | "
              f"avg summary {sum(sums)//len(sums):4d} words | "
              f"compression ~{avg_compression:.1f}x")

    # Preview one sample
    print("\n🔍 Sample preview (train[0]):")
    sample = ds["train"][0]
    print(f"  TITLE:   {sample['title']}")
    print(f"  TEXT:    {sample['text'][:200]}...")
    print(f"  SUMMARY: {sample['summary'][:200]}...")


if __name__ == "__main__":
    main()
