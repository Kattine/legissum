# 🏛️ LegisSum — Fine-tuning LLMs to Make U.S. Congressional Bills Readable

> Mini Hackathon #3 | Generative AI | LoRA Fine-tuning

A targeted fine-tuning project that teaches a small LLM to summarize dense U.S. Congressional legislation into plain English — demonstrating a clear **before/after capability jump** using QLoRA on the BillSum dataset.

---

## 📌 Problem Statement

U.S. Congressional bills are notoriously difficult to read. A base LLM either hallucinates, copies text verbatim, or produces incoherent output when given raw legislative text. After fine-tuning on 16,000+ bill-summary pairs, the model learns to produce concise, citizen-readable summaries matching the style of official Congressional Research Service (CRS) summaries.

---

## 🗂️ Project Structure

```
legissum/
├── data/                        # Downloaded & processed datasets
│   ├── raw/                     # Original BillSum JSONL (auto-downloaded)
│   └── processed/               # Filtered + formatted training data
│
├── src/                         # Core pipeline scripts
│   ├── 01_download_data.py      # Download & explore BillSum
│   ├── 02_preprocess.py         # Filter + convert to instruction format
│   ├── 03_train.py              # QLoRA fine-tuning with TRL/PEFT
│   ├── 04_evaluate.py           # ROUGE scoring + before/after comparison
│   └── utils.py                 # Shared helpers
│
├── notebooks/
│   └── demo.ipynb               # Colab-ready end-to-end demo
│
├── outputs/                     # Generated summaries, eval results
├── models/                      # Saved LoRA adapters (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local VSCode)

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/legissum.git
cd legissum
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download & preprocess data
```bash
python src/01_download_data.py
python src/02_preprocess.py
```

### 3. Train
```bash
python src/03_train.py
```

### 4. Evaluate & see Before/After
```bash
python src/04_evaluate.py
```

---

## ☁️ Quick Start (Google Colab / Kaggle — No GPU needed locally)

Open `notebooks/demo.ipynb` directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/legissum/blob/main/notebooks/demo.ipynb)

Free T4 GPU on Colab is sufficient for 2-epoch QLoRA training (~3–4 hours).

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [FiscalNote/billsum](https://huggingface.co/datasets/FiscalNote/billsum) |
| Train samples | ~16,000 (after filtering) |
| Test samples | ~2,800 |
| License | CC0-1.0 (fully free) |
| Input | U.S. Congressional bill text |
| Output | Official CRS summary (human-written) |
| Avg compression ratio | ~10–15x |

---

## 🤖 Model & Training

| Property | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Method | QLoRA (4-bit quantization + LoRA adapters) |
| LoRA rank | r=16, alpha=32 |
| Trainable params | ~0.2% of total |
| Epochs | 2 |
| Batch size | 2 (grad accum 4 → effective 8) |
| Hardware | Single GPU (T4 16GB / RTX 3080+) |

---

## 📈 Results

| Metric | Base Model | Fine-tuned |
|---|---|---|
| ROUGE-1 | ~0.18 | ~0.45 |
| ROUGE-2 | ~0.06 | ~0.22 |
| ROUGE-L | ~0.15 | ~0.38 |

---

## ⚠️ Ethics & Risks

- **Hallucination risk**: The model may generate plausible-sounding but factually incorrect summaries. All outputs should be human-reviewed before use in civic contexts.
- **Bias in source data**: BillSum covers bills up to ~2018. Recent legislative language or new policy domains may be underrepresented.
- **Misuse potential**: Automated bill summarization could be weaponized to produce misleading framings. Evaluation should include adversarial testing.
- **Evaluation gap**: ROUGE scores measure n-gram overlap but not factual accuracy — a key limitation for legal summarization.

---

## 📚 Citation

```bibtex
@inproceedings{kornilova-eidelman-2019-billsum,
    title     = "BillSum: A Corpus for Automatic Summarization of US Legislation",
    author    = "Kornilova, Anastassia and Eidelman, Vladimir",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    year      = "2019",
}
```
