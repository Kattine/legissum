"""
bertscore_samples.py
────────────────────
Run BERTScore on the existing sample_comparisons.json (no model loading needed).
"""

import json
import os
from bert_score import score as bert_score_fn

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

with open(os.path.join(OUTPUTS_DIR, "sample_comparisons.json")) as f:
    samples = json.load(f)

refs     = [s["reference"]  for s in samples]
base     = [s["base_model"] for s in samples]
finetuned = [s["finetuned"] for s in samples]

print(f"Scoring {len(samples)} samples with BERTScore…\n")

P_b, R_b, F_b = bert_score_fn(base,      refs, lang="en", verbose=False)
P_f, R_f, F_f = bert_score_fn(finetuned, refs, lang="en", verbose=False)

scores_base = {"precision": round(P_b.mean().item(), 4),
               "recall":    round(R_b.mean().item(), 4),
               "f1":        round(F_b.mean().item(), 4)}
scores_ft   = {"precision": round(P_f.mean().item(), 4),
               "recall":    round(R_f.mean().item(), 4),
               "f1":        round(F_f.mean().item(), 4)}

print(f"{'Metric':<12} {'Base Model':>12} {'Fine-tuned':>12} {'Δ':>10}")
print("-" * 50)
for k in ["precision", "recall", "f1"]:
    delta = scores_ft[k] - scores_base[k]
    print(f"{k:<12} {scores_base[k]:>12.4f} {scores_ft[k]:>12.4f} {delta:>+10.4f}")

out = os.path.join(OUTPUTS_DIR, "bertscore_scores.json")
with open(out, "w") as f:
    json.dump({"base": scores_base, "finetuned": scores_ft}, f, indent=2)

print(f"\n✅ Saved to outputs/bertscore_scores.json")
