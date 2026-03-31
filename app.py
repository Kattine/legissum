"""
app.py — LegisSum Gradio Demo
──────────────────────────────────────────────────────────────────────────────
Interactive demo for the LegisSum fine-tuned bill summarizer.

Run:
    python app.py

Tabs:
  1. Summarize a Bill  — paste any bill text and get a plain-English summary
  2. Before vs After   — side-by-side comparison of base vs fine-tuned model
                         (loads from outputs/sample_comparisons.json if present)
"""

import json
import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional PEFT import — only needed when adapter weights exist
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ADAPTER_PATH   = ROOT / "models" / "billsum-lora"
OUTPUTS_DIR    = ROOT / "outputs"
SAMPLES_FILE   = OUTPUTS_DIR / "sample_comparisons.json"
ROUGE_FILE     = OUTPUTS_DIR / "rouge_scores.json"

BASE_MODEL_ID  = os.environ.get("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
HF_ADAPTER_ID  = os.environ.get("HF_MODEL_ID", "")   # e.g. "yourname/legissum-lora"

SYSTEM_INSTRUCTION = (
    "You are a legislative assistant helping everyday citizens understand U.S. law. "
    "Read the following Congressional bill and write a clear, concise summary. "
    "Focus on: what the bill does, who it affects, and its key provisions. "
    "Use plain English — no legal jargon."
)

# ── Sample bills for quick demo ────────────────────────────────────────────────
SAMPLE_BILLS = {
    "Infrastructure example": (
        "H.R. 3684 — Infrastructure Investment and Jobs Act\n\n"
        "This Act provides for the improvement of the nation's infrastructure, "
        "including roads, bridges, broadband internet, clean water systems, and "
        "passenger and freight rail. The bill authorizes $550 billion in new federal "
        "spending over five years. Funds are allocated to the Department of "
        "Transportation, the Environmental Protection Agency, and other federal "
        "agencies. States must submit plans for spending approval. Projects must "
        "comply with Buy America requirements, ensuring that iron, steel, and "
        "manufactured products are produced in the United States."
    ),
    "Healthcare example": (
        "S. 1234 — Affordable Prescription Drug Act\n\n"
        "This Act authorizes the Secretary of Health and Human Services to negotiate "
        "directly with pharmaceutical manufacturers to lower the prices of "
        "prescription drugs covered under Medicare Part D. The bill caps "
        "out-of-pocket drug costs for Medicare beneficiaries at $2,000 per year. "
        "Manufacturers who raise prices faster than inflation must pay rebates to "
        "the federal government. The Congressional Budget Office estimates this "
        "will reduce federal drug spending by $288 billion over 10 years."
    ),
    "Education example": (
        "H.R. 5678 — Student Loan Forgiveness Act\n\n"
        "This Act establishes a program for the cancellation of outstanding federal "
        "student loan debt for borrowers who have made payments for at least 20 years "
        "under an income-driven repayment plan. Eligible borrowers include those with "
        "undergraduate and graduate loans issued by the Department of Education. "
        "Borrowers employed in public service jobs may qualify after 10 years of "
        "payments. The Department of Education must provide written notice to all "
        "eligible borrowers within 180 days of enactment."
    ),
}


# ── Model loading (lazy, cached) ───────────────────────────────────────────────
_model = None
_tokenizer = None
_model_label = ""


def load_model():
    global _model, _tokenizer, _model_label

    if _model is not None:
        return _model, _tokenizer, _model_label

    print(f"\n⬇️  Loading base model: {BASE_MODEL_ID}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    # Try to load LoRA adapters: local first, then HF Hub
    if PEFT_AVAILABLE and ADAPTER_PATH.exists():
        _model = PeftModel.from_pretrained(_model, str(ADAPTER_PATH))
        _model_label = "Fine-tuned (LegisSum — local adapter)"
        print(f"✅ Loaded local LoRA adapters from {ADAPTER_PATH}")
    elif PEFT_AVAILABLE and HF_ADAPTER_ID:
        _model = PeftModel.from_pretrained(_model, HF_ADAPTER_ID)
        _model_label = f"Fine-tuned (LegisSum — {HF_ADAPTER_ID})"
        print(f"✅ Loaded LoRA adapters from HF Hub: {HF_ADAPTER_ID}")
    else:
        _model_label = "⚠️ Base model only (run training first to load fine-tuned adapter)"
        print("⚠️  No LoRA adapter found — using base model only")

    return _model, _tokenizer, _model_label


def generate_summary(model, tokenizer, bill_text: str, max_new_tokens: int = 300) -> str:
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
    if "### Summary:" in decoded:
        return decoded.split("### Summary:")[-1].strip()
    return decoded.strip()


# ── Tab 1: Summarize ───────────────────────────────────────────────────────────

def summarize_bill(bill_text: str, progress=gr.Progress()):
    if not bill_text.strip():
        return "Please paste bill text above.", ""

    progress(0.1, desc="Loading model...")
    model, tokenizer, label = load_model()

    progress(0.4, desc="Generating summary...")
    summary = generate_summary(model, tokenizer, bill_text)

    progress(1.0, desc="Done!")
    model_info = f"**Model:** {label}"
    return summary, model_info


def load_sample(sample_name: str) -> str:
    return SAMPLE_BILLS.get(sample_name, "")


# ── Tab 2: Before vs After ─────────────────────────────────────────────────────

def load_comparisons():
    if SAMPLES_FILE.exists():
        with open(SAMPLES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data
    return []


def load_rouge_scores():
    if ROUGE_FILE.exists():
        with open(ROUGE_FILE) as f:
            scores = json.load(f)
        base = scores.get("base", {})
        ft   = scores.get("finetuned", {})
        lines = [
            "### ROUGE Score Comparison (50 test samples)\n",
            "| Metric | Base Model | Fine-tuned | Improvement |",
            "|--------|-----------|------------|-------------|",
        ]
        for k in ["rouge1", "rouge2", "rougeL"]:
            b, f_ = base.get(k, 0), ft.get(k, 0)
            delta = f_ - b
            lines.append(f"| {k} | {b:.4f} | {f_:.4f} | {delta:+.4f} |")
        return "\n".join(lines)
    return "_No ROUGE scores found. Run `python src/04_evaluate.py` after training._"


def show_comparison(index: int):
    comparisons = load_comparisons()
    if not comparisons:
        msg = "_No sample comparisons found. Run `python src/04_evaluate.py` after training to generate them._"
        return msg, msg, msg, ""
    idx = max(0, min(index, len(comparisons) - 1))
    item = comparisons[idx]
    title = f"**Bill:** {item.get('title', 'N/A')}"
    return (
        item.get("base_model", ""),
        item.get("finetuned", ""),
        item.get("reference", ""),
        title,
    )


# ── Build UI ───────────────────────────────────────────────────────────────────

def build_app():
    comparisons = load_comparisons()
    n_samples = len(comparisons)

    with gr.Blocks(title="LegisSum — Congressional Bill Summarizer", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
# LegisSum — Congressional Bill Summarizer
**Fine-tuned Mistral-7B via QLoRA on 16,000+ Congressional bill/summary pairs**

Makes U.S. legislation readable for everyday citizens.
            """
        )

        with gr.Tabs():

            # ── Tab 1: Summarize ──────────────────────────────────────────────
            with gr.Tab("Summarize a Bill"):
                gr.Markdown("Paste any Congressional bill text below, or load a sample to try it out.")

                with gr.Row():
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_BILLS.keys()),
                        label="Load a sample bill",
                        value=None,
                    )
                    load_btn = gr.Button("Load Sample", variant="secondary")

                bill_input = gr.Textbox(
                    label="Bill Text",
                    placeholder="Paste bill title and text here...",
                    lines=12,
                    max_lines=30,
                )
                char_count = gr.Markdown("_0 characters_")

                summarize_btn = gr.Button("Summarize", variant="primary", size="lg")

                summary_output = gr.Textbox(
                    label="Plain-English Summary",
                    lines=8,
                    interactive=False,
                )
                model_info_out = gr.Markdown()

                # Wire up events
                load_btn.click(fn=load_sample, inputs=sample_dropdown, outputs=bill_input)
                bill_input.change(
                    fn=lambda t: f"_{len(t):,} characters_",
                    inputs=bill_input,
                    outputs=char_count,
                )
                summarize_btn.click(
                    fn=summarize_bill,
                    inputs=bill_input,
                    outputs=[summary_output, model_info_out],
                )

            # ── Tab 2: Before vs After ────────────────────────────────────────
            with gr.Tab("Before vs After Fine-Tuning"):
                gr.Markdown(
                    "Compare what the **base model** produces vs the **fine-tuned LegisSum model** "
                    "on held-out test examples. These outputs were generated during evaluation."
                )

                if n_samples == 0:
                    gr.Markdown(
                        "> **No comparison data yet.** "
                        "Run `python src/04_evaluate.py` after training to generate `outputs/sample_comparisons.json`."
                    )
                else:
                    example_slider = gr.Slider(
                        minimum=0,
                        maximum=n_samples - 1,
                        step=1,
                        value=0,
                        label=f"Example (0 – {n_samples - 1})",
                    )
                    bill_title = gr.Markdown()

                    with gr.Row():
                        base_out = gr.Textbox(label="Base Model (Before)", lines=10, interactive=False)
                        ft_out   = gr.Textbox(label="Fine-tuned LegisSum (After)", lines=10, interactive=False)
                        ref_out  = gr.Textbox(label="Reference Summary (Ground Truth)", lines=10, interactive=False)

                    example_slider.change(
                        fn=show_comparison,
                        inputs=example_slider,
                        outputs=[base_out, ft_out, ref_out, bill_title],
                    )
                    # Load first example on startup
                    demo.load(
                        fn=lambda: show_comparison(0),
                        outputs=[base_out, ft_out, ref_out, bill_title],
                    )

                gr.Markdown("---")
                gr.Markdown(load_rouge_scores())

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
