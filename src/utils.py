"""
utils.py — Shared helpers across the pipeline
"""

import os
import json
import random
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} rows → {path}")


def preview(data: List[Dict], n: int = 3, seed: int = 42) -> None:
    """Print n random samples from a dataset list."""
    random.seed(seed)
    for item in random.sample(data, min(n, len(data))):
        print("─" * 60)
        for k, v in item.items():
            val = str(v)
            print(f"  {k}: {val[:300]}{'...' if len(val) > 300 else ''}")
    print()


def compression_ratio(data: List[Dict],
                       input_key: str = "input",
                       output_key: str = "output") -> float:
    inputs  = [len(d[input_key].split())  for d in data]
    outputs = [len(d[output_key].split()) for d in data]
    return (sum(inputs) / len(inputs)) / (sum(outputs) / len(outputs))
