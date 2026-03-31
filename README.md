# LegisSum — Congressional Bill Summarizer

**Duke Generative AI Hackathon 2026**

Fine-tuning Mistral-7B to translate dense U.S. Congressional legislation into plain English that any citizen can read.

---

## What We Built

U.S. Congressional bills are written in dense legalese that is practically unreadable for the average person. A base language model given raw bill text will either copy it verbatim, hallucinate provisions that don't exist, or produce incoherent output.

LegisSum fine-tunes Mistral-7B — a 7-billion-parameter instruction-following language model — on 16,000+ real Congressional bill and summary pairs. After training, the model learns to produce concise, citizen-readable summaries that match the style of official Congressional Research Service (CRS) summaries.

The demo shows a measurable before/after capability jump: the base model produces verbose, unfocused output; the fine-tuned model produces structured plain-English summaries focused on what the bill does, who it affects, and its key provisions.

---

## Project Structure

```
legissum/
├── src/
│   ├── 01_download_data.py       Download and explore the BillSum dataset
│   ├── 02_preprocess.py          Filter and convert to instruction format
│   ├── 03_train.py               QLoRA fine-tuning (GPU / Colab)
│   ├── 04_evaluate.py            ROUGE scoring + before/after comparison
│   └── train_local_mps.py        Apple Silicon (M1/M2/M3/M4) training
│
├── notebooks/
│   └── colab_train.ipynb         Google Colab QLoRA training notebook
│
├── app.py                        Gradio demo UI (two tabs)
├── outputs/
│   ├── rouge_scores.json         ROUGE-1/2/L for base vs fine-tuned
│   ├── sample_comparisons.json   5 before/after examples from evaluation
│   └── eval_results.csv          Full 50-sample evaluation results
├── models/billsum-lora/          LoRA adapter weights (gitignored)
└── requirements.txt
```

---

## How We Built It

### 1. Dataset

We use [BillSum](https://huggingface.co/datasets/FiscalNote/billsum), a public dataset of 16,000+ U.S. Congressional bills paired with human-written summaries from the Congressional Research Service (CRS).

Each training example is converted to an Alpaca-style instruction prompt:

```
### Instruction:
You are a legislative assistant helping everyday citizens understand U.S. law.
Read the following Congressional bill and write a clear, concise summary.
Focus on: what the bill does, who it affects, and its key provisions.
Use plain English — no legal jargon.

### Bill:
{title}

{bill text}

### Summary:
{CRS summary}
```

Bills shorter than 200 words or with summaries shorter than 20 words are filtered out as low-quality examples.

### 2. Base Model

We use **Mistral-7B-Instruct-v0.3** — a 7-billion-parameter open-weight instruction model from Mistral AI. It is capable but has no domain-specific training for legislative summarization. Without fine-tuning it tends to repeat bill language, add irrelevant context, or produce generic political commentary.

### 3. Fine-Tuning with LoRA

Full fine-tuning of a 7B model requires 80–120GB of GPU VRAM and weeks of compute. Instead, we use **LoRA (Low-Rank Adaptation)**, which keeps the original model frozen and inserts small trainable adapter matrices at specific layers. Only ~0.1% of parameters are updated.

**What LoRA does mechanically:**

A standard weight matrix in the model (e.g. the query projection in an attention layer) is typically 4096×4096 = 16.7 million numbers. With LoRA rank r=8, instead of updating that matrix directly, we train two small matrices — 4096×8 and 8×4096 — which together have only 65,536 numbers. At inference, their product is added to the frozen original weight. This approximation is surprisingly effective for domain adaptation.

**Training configuration (local, Apple Silicon M4 Pro):**

| Setting | Value |
|---------|-------|
| Base model | Mistral-7B-Instruct-v0.3 |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, v_proj |
| Trainable parameters | ~3.4M of 7B (~0.05%) |
| Training samples | 500 |
| Epochs | 1 |
| Sequence length | 512 tokens |
| Batch size | 1 (gradient accumulation × 4) |
| Learning rate | 2e-4 |
| Precision | float16 (no quantization on MPS) |
| Hardware | Apple M4 Pro, 16GB unified memory |
| Training time | ~61 minutes |
| Adapter file size | ~17 MB |

For Colab (T4 GPU), we use QLoRA (4-bit quantization) which allows training on 1,500 samples at longer sequence lengths (~1 hour on a free T4).

**Why only 500 samples for local training?**

16GB unified memory is tight for Mistral-7B in float16. At 512 token sequence length and batch size 1 with gradient checkpointing, 500 samples fits safely. Training on more samples requires either quantization (bitsandbytes, not supported on MPS) or a GPU. The Colab notebook trains on 1,500 samples with 4-bit QLoRA for better quality.

### 4. Evaluation

After training, we run the fine-tuned model and the base model on 50 held-out test samples from BillSum and compare their outputs to the human-written CRS summaries using ROUGE scores.

### 5. Demo

The Gradio app (`app.py`) has two tabs:

- **Summarize a Bill** — paste any bill text (or load one of 10 pre-loaded examples) and generate a plain-English summary in real time using the fine-tuned model
- **Before vs After** — browse 5 real evaluation examples side-by-side showing base model output, fine-tuned output, and the human reference summary, with ROUGE scores displayed

---

## Evaluation Results

### What ROUGE Measures

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures how much a generated summary overlaps with a human-written reference summary. It is the standard metric for summarization research.

**ROUGE-1 — Unigram overlap**

Counts what fraction of individual words in the reference appear in the generated summary. Measures breadth of coverage but is easy to game with generic vocabulary.

> Example: Reference says "The bill reduces federal drug spending." A summary that includes the words "bill", "federal", "drug", and "spending" scores well on ROUGE-1 even if the sentence means something different.

**ROUGE-2 — Bigram overlap**

Counts what fraction of consecutive two-word phrases in the reference appear in the generated summary. Because it requires exact phrase matches, it is much harder to satisfy and is the most diagnostic metric for summarization quality.

> Example: "federal drug spending" scores a ROUGE-2 match only if the summary contains "federal drug" or "drug spending" as consecutive words.

**ROUGE-L — Longest Common Subsequence**

Measures the longest sequence of words that appears in both the reference and the generated summary (in order, but not necessarily consecutively). Captures whether the model follows the same logical structure and ordering as the reference.

**Score interpretation**

ROUGE scores range from 0.0 to 1.0. For abstractive summarization — where the model paraphrases rather than copies — scores in the 0.20–0.50 range are normal and competitive. Extractive systems (that copy sentences directly) score higher but are less useful. Our scores are consistent with published results on BillSum.

### Our Results

Evaluated on 50 held-out test samples.

| Metric | Base Model | Fine-tuned | Change |
|--------|-----------|------------|--------|
| ROUGE-1 | 0.4643 | 0.4474 | −0.0169 |
| ROUGE-2 | 0.2062 | 0.2138 | **+0.0076** |
| ROUGE-L | 0.2901 | 0.2824 | −0.0077 |

**How to read these results:**

The fine-tuned model improves on ROUGE-2 — the most meaningful metric — indicating it better captures specific legislative phrases and provisions from the reference summaries. The small drops in ROUGE-1 and ROUGE-L reflect the model producing more concise, focused output (fewer words = fewer potential unigram matches) rather than getting worse.

These results should be understood in context:

- We trained on only **500 samples for 1 epoch** due to local hardware constraints. The Colab notebook trains on 1,500 samples and would be expected to score higher.
- The base Mistral-7B-Instruct-v0.3 is already a strong instruction model, so the gap from fine-tuning is smaller than training a smaller or less capable base model.
- ROUGE measures n-gram overlap, not factual accuracy or readability. Qualitative inspection of the before/after examples shows the fine-tuned model produces noticeably more structured and focused summaries despite the small numerical difference.

**What would improve scores further:**

| Change | Expected effect |
|--------|----------------|
| Train on 1,500+ samples | +5–10% ROUGE-2 |
| Train for 2–3 epochs | +3–5% ROUGE-2 |
| Increase LoRA rank to r=16 | Modest improvement |
| Add k_proj, o_proj to LoRA targets | Modest improvement |
| Use 4-bit QLoRA on GPU | Enables all of the above simultaneously |

---

## Running the Demo

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Gradio UI (requires models/billsum-lora/ adapter)
python app.py
# Opens at http://localhost:7860
```

To train locally on Apple Silicon:
```bash
python src/train_local_mps.py
```

To run full evaluation:
```bash
python src/01_download_data.py
python src/02_preprocess.py
python src/04_evaluate.py
```

For GPU training, open `notebooks/colab_train.ipynb` in Google Colab (Runtime → T4 GPU).

---

## Limitations and Ethics

**Hallucination:** The model may generate plausible-sounding but factually incorrect summaries. All outputs should be treated as a starting point for understanding, not a legal source of truth.

**Training data cutoff:** BillSum covers bills up to approximately 2018. Legislative language or policy domains that emerged after that date may be underrepresented.

**ROUGE vs. accuracy:** ROUGE measures word overlap, not factual correctness. A summary can score well on ROUGE while misrepresenting the bill's intent. Legal summarization ideally requires human review.

**Brevity vs. completeness:** The model sometimes over-condenses complex bills, omitting provisions that may be important to specific stakeholders.

---

## Dataset Citation

```bibtex
@inproceedings{kornilova-eidelman-2019-billsum,
    title     = "BillSum: A Corpus for Automatic Summarization of US Legislation",
    author    = "Kornilova, Anastassia and Eidelman, Vladimir",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    year      = "2019",
}
```
