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
    "Infrastructure — Roads & Broadband": (
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
    "Healthcare — Drug Price Negotiation": (
        "S. 1234 — Affordable Prescription Drug Act\n\n"
        "This Act authorizes the Secretary of Health and Human Services to negotiate "
        "directly with pharmaceutical manufacturers to lower the prices of "
        "prescription drugs covered under Medicare Part D. The bill caps "
        "out-of-pocket drug costs for Medicare beneficiaries at $2,000 per year. "
        "Manufacturers who raise prices faster than inflation must pay rebates to "
        "the federal government. The Congressional Budget Office estimates this "
        "will reduce federal drug spending by $288 billion over 10 years."
    ),
    "Education — Student Loan Relief": (
        "H.R. 5678 — Student Loan Forgiveness Act\n\n"
        "This Act establishes a program for the cancellation of outstanding federal "
        "student loan debt for borrowers who have made payments for at least 20 years "
        "under an income-driven repayment plan. Eligible borrowers include those with "
        "undergraduate and graduate loans issued by the Department of Education. "
        "Borrowers employed in public service jobs may qualify after 10 years of "
        "payments. The Department of Education must provide written notice to all "
        "eligible borrowers within 180 days of enactment."
    ),
    "Climate — Clean Energy Investment": (
        "S. 2112 — Clean Energy Transition and Jobs Act\n\n"
        "This Act establishes a Clean Energy Accelerator within the Department of "
        "Energy to provide grants and low-interest loans for the deployment of "
        "solar, wind, geothermal, and battery storage technologies. The bill "
        "requires electric utilities to obtain 80 percent of their electricity from "
        "clean energy sources by 2035. It creates a Clean Energy Manufacturing Tax "
        "Credit for domestic production of solar panels, wind turbines, and electric "
        "vehicle batteries. The Act also establishes a Civilian Climate Corps to "
        "employ 300,000 workers in conservation, resilience, and clean energy "
        "projects on public lands and coastal areas."
    ),
    "Veterans — Mental Health Services": (
        "H.R. 4421 — Veterans Mental Health Care Improvement Act\n\n"
        "This Act directs the Secretary of Veterans Affairs to expand mental health "
        "services for veterans, including same-day access to mental health care at "
        "all VA medical centers. The bill authorizes $500 million annually to hire "
        "additional licensed mental health providers and peer support specialists. "
        "It requires the VA to screen all veterans for military sexual trauma, "
        "post-traumatic stress disorder, and traumatic brain injury. Veterans in "
        "rural areas must be provided telehealth services within 30 days of a "
        "request. The Act establishes a suicide prevention hotline staffed by "
        "licensed clinicians around the clock."
    ),
    "Immigration — Border Processing": (
        "H.R. 6789 — Border Security and Humanitarian Processing Act\n\n"
        "This Act authorizes the hiring of 2,000 additional U.S. Customs and Border "
        "Protection officers and 1,500 immigration judges to reduce case backlogs. "
        "The bill establishes Humanitarian Processing Centers at ports of entry where "
        "asylum seekers may present claims in an orderly and safe manner. It requires "
        "asylum applications to be adjudicated within 180 days. The Act provides "
        "$2.5 billion for technology upgrades including biometric screening and "
        "vehicle inspection equipment. Unaccompanied minors must be placed with "
        "vetted sponsors within 72 hours of arrival."
    ),
    "Technology — Data Privacy": (
        "S. 3301 — American Data Privacy and Protection Act\n\n"
        "This Act establishes a federal baseline for data privacy, requiring "
        "companies that collect personal data on more than 100,000 individuals to "
        "provide clear notice and obtain opt-in consent before sharing sensitive "
        "data with third parties. The Federal Trade Commission is granted authority "
        "to enforce violations with civil penalties of up to $50,000 per day per "
        "violation. The bill bans the sale of precise geolocation and health data "
        "without explicit consent. Individuals have the right to access, correct, "
        "and delete their personal data. Children under 16 are granted heightened "
        "protections and companies may not use algorithmic targeting toward minors."
    ),
    "Criminal Justice — Sentencing Reform": (
        "H.R. 7890 — Fair Sentencing Modernization Act\n\n"
        "This Act reduces mandatory minimum sentences for certain non-violent drug "
        "offenses and expands the safety valve provision to allow judges discretion "
        "in sentencing first-time, low-level offenders. The bill makes retroactive "
        "the reduction in the crack-to-powder cocaine sentencing disparity enacted "
        "in 2010. It establishes a federal Second Chance Grant Program providing "
        "$200 million annually to state and local reentry programs including job "
        "training, housing assistance, and substance abuse treatment. Federal "
        "prisoners who complete education or vocational training programs may earn "
        "up to 12 months of early release credit."
    ),
    "Finance — Federal Forage Fee (Eval Set)": (
        "Federal Forage Fee Act of 1993\n\n"
        "SECTION 1. SHORT TITLE.\n"
        "This Act may be cited as the 'Federal Forage Fee Act of 1993'.\n\n"
        "SECTION 2. FINDINGS AND PURPOSE.\n"
        "The Congress finds that — (1) the public lands of the United States contain "
        "valuable forage resources; (2) in many areas of the West, public and private "
        "lands are interdependent for forage, water, and wildlife habitat; (3) Federal "
        "grazing fees have not kept pace with the fair market value of forage on "
        "private lands.\n\n"
        "SECTION 3. FORAGE FEES.\n"
        "(a) All grazing operations conducted on Federal land shall be subject to "
        "applicable Federal, State, and local environmental and land use requirements. "
        "(b) The Secretary of Agriculture and the Secretary of the Interior shall "
        "establish a forage fee formula based on fair market value of forage, "
        "assessed on a per animal unit month basis, applicable to lands under "
        "their respective jurisdictions."
    ),
    "Awards — Merchant Marine Gold Medal (Eval Set)": (
        "Merchant Marine of World War II Congressional Gold Medal Act\n\n"
        "SECTION 1. SHORT TITLE.\n"
        "This Act may be cited as the 'Merchant Marine of World War II Congressional "
        "Gold Medal Act'.\n\n"
        "SECTION 2. FINDINGS.\n"
        "The Congress finds the following: (1) 2015 marks the 70th anniversary of "
        "the Allied victory in World War II. (2) The United States Merchant Marine "
        "was integral in providing the link between domestic production and the "
        "fighting forces overseas, supplying combat equipment, fuel, food, "
        "commodities, and raw materials. (3) Fleet Admiral Ernest J. King acknowledged "
        "the indispensability of the Merchant Marine. (4) The Merchant Marine suffered "
        "a higher per capita casualty rate than any other U.S. service.\n\n"
        "SECTION 3. CONGRESSIONAL GOLD MEDAL.\n"
        "The Speaker of the House of Representatives and the President pro tempore "
        "of the Senate shall make appropriate arrangements for the award, on behalf "
        "of Congress, of a single gold medal to the United States Merchant Marine of "
        "World War II. Following its award, the medal shall be given to the American "
        "Merchant Marine Museum for display."
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

    css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --navy:      #080d1a;
    --navy-mid:  #0e1628;
    --navy-card: #111b2e;
    --gold:      #c9a84c;
    --gold-dim:  #9b7c36;
    --cream:     #f0e8d5;
    --text:      #d8cfc0;
    --muted:     #6b7a99;
    --border:    rgba(201,168,76,0.18);
    --glow:      rgba(201,168,76,0.07);
}

/* ── Base ── */
body, .gradio-container {
    background: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    min-height: 100vh;
}
.gradio-container {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(201,168,76,0.08) 0%, transparent 70%),
        var(--navy) !important;
}

/* ── Header / hero ── */
.hero-wrap { text-align: center; padding: 2.5rem 1rem 1rem; }
.hero-wrap h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2rem, 5vw, 3.4rem) !important;
    font-weight: 900 !important;
    color: var(--cream) !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 .4rem;
}
.hero-wrap .eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: .6rem;
}
.hero-wrap p {
    color: var(--muted) !important;
    font-size: .95rem;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.6;
}
.rule {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.2rem auto;
}

/* ── Tabs ── */
.tabs { border: none !important; background: transparent !important; }
.tab-nav { border-bottom: 1px solid var(--border) !important; background: transparent !important; padding: 0 !important; }
.tab-nav button {
    font-family: 'DM Mono', monospace !important;
    font-size: .75rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: .8rem 1.4rem !important;
    transition: all .2s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── Cards / panels ── */
.gr-box, .gr-form, .block, .panel {
    background: var(--navy-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Textareas & inputs ── */
textarea, input[type=text] {
    background: var(--navy-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .85rem !important;
    line-height: 1.65 !important;
    transition: border-color .2s !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
    outline: none !important;
}
label span, .label-wrap span {
    font-family: 'DM Mono', monospace !important;
    font-size: .7rem !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    color: var(--gold-dim) !important;
}

/* ── Primary button ── */
button.primary, button[variant=primary], .primary-btn {
    background: var(--gold) !important;
    color: #0a0f1a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .75rem !important;
    font-weight: 500 !important;
    letter-spacing: .2em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: .75rem 2rem !important;
    cursor: pointer !important;
    transition: all .2s !important;
    box-shadow: 0 0 24px rgba(201,168,76,0.2) !important;
}
button.primary:hover { background: #dbb95a !important; box-shadow: 0 0 32px rgba(201,168,76,0.35) !important; }

/* ── Secondary button ── */
button.secondary, button[variant=secondary] {
    background: transparent !important;
    color: var(--gold) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .72rem !important;
    letter-spacing: .15em !important;
    text-transform: uppercase !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: .6rem 1.2rem !important;
    transition: all .2s !important;
}
button.secondary:hover { border-color: var(--gold) !important; background: var(--glow) !important; }

/* ── Dropdown ── */
.gr-dropdown, select {
    background: var(--navy-mid) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 4px !important;
}

/* ── Slider ── */
input[type=range] { accent-color: var(--gold) !important; }

/* ── Markdown output ── */
.md p, .md li { color: var(--text) !important; font-size: .93rem !important; line-height: 1.7 !important; }
.md h3 { font-family: 'Playfair Display', serif !important; color: var(--cream) !important; }
.md table { border-collapse: collapse; width: 100%; }
.md th {
    font-family: 'DM Mono', monospace !important;
    font-size: .68rem !important;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--gold-dim) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: .5rem .75rem !important;
    background: transparent !important;
}
.md td { padding: .45rem .75rem !important; border-bottom: 1px solid rgba(201,168,76,0.07) !important; font-size: .85rem !important; color: var(--text) !important; }
.md td:first-child { font-family: 'DM Mono', monospace !important; color: var(--gold) !important; }

/* ── Character count / info text ── */
.char-count p { font-family: 'DM Mono', monospace !important; font-size: .7rem !important; color: var(--muted) !important; }

/* ── Progress bar ── */
.progress-bar { background: var(--gold) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Footer stamp ── */
.stamp {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    padding: 1rem 0 2rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
"""

    with gr.Blocks(title="LegisSum — Congressional Bill Summarizer", css=css) as demo:

        gr.HTML("""
<div class="hero-wrap">
  <div class="eyebrow">U.S. Legislative Intelligence</div>
  <h1>LegisSum</h1>
  <div class="rule"></div>
  <p>Fine-tuned Mistral-7B on 16,000+ Congressional bill/summary pairs.<br>
     Plain English. No legal jargon.</p>
</div>
""")

        with gr.Tabs():

            # ── Tab 1: Summarize ──────────────────────────────────────────────
            with gr.Tab("⬡  Summarize a Bill"):
                gr.Markdown(
                    "<div style='font-family:DM Mono,monospace;font-size:.72rem;letter-spacing:.14em;"
                    "text-transform:uppercase;color:#6b7a99;padding:.6rem 0 1rem'>"
                    "Paste any Congressional bill text below, or load a sample to try it out.</div>"
                )

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
                char_count = gr.Markdown(
                    "_0 characters_",
                    elem_classes=["char-count"],
                )

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
            with gr.Tab("⬡  Before vs After"):
                gr.Markdown(
                    "<div style='font-family:DM Mono,monospace;font-size:.72rem;letter-spacing:.14em;"
                    "text-transform:uppercase;color:#6b7a99;padding:.6rem 0 1rem'>"
                    "Base model vs fine-tuned LegisSum — held-out test examples from evaluation.</div>"
                )

                # ── ROUGE score summary cards ──────────────────────────────────
                gr.HTML("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:0 0 1.5rem">
  <div style="background:#111b2e;border:1px solid rgba(201,168,76,0.18);border-radius:4px;padding:1.2rem;text-align:center">
    <div style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:#9b7c36;margin-bottom:.5rem">ROUGE-1 · Unigram</div>
    <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#f0e8d5;font-weight:700">0.4474</div>
    <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:#6b7a99;margin-top:.3rem">base: 0.4643</div>
  </div>
  <div style="background:#111b2e;border:1px solid rgba(201,168,76,0.35);border-radius:4px;padding:1.2rem;text-align:center;box-shadow:0 0 20px rgba(201,168,76,0.08)">
    <div style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:#c9a84c;margin-bottom:.5rem">ROUGE-2 · Bigram ↑</div>
    <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#c9a84c;font-weight:700">0.2138</div>
    <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:#6b7a99;margin-top:.3rem">base: 0.2062 &nbsp;·&nbsp; <span style="color:#c9a84c">+0.0076</span></div>
  </div>
  <div style="background:#111b2e;border:1px solid rgba(201,168,76,0.18);border-radius:4px;padding:1.2rem;text-align:center">
    <div style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:#9b7c36;margin-bottom:.5rem">ROUGE-L · Longest Match</div>
    <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#f0e8d5;font-weight:700">0.2824</div>
    <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:#6b7a99;margin-top:.3rem">base: 0.2901</div>
  </div>
</div>
<div style="font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.1em;color:#6b7a99;text-align:center;margin-bottom:1.5rem">
  Evaluated on 50 held-out test samples &nbsp;·&nbsp; 500 training samples &nbsp;·&nbsp; 1 epoch &nbsp;·&nbsp; LoRA r=8
</div>
""")

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
                        label=f"Example (0–{n_samples - 1})  ·  Use slider to browse evaluation examples",
                    )
                    bill_title = gr.Markdown()

                    with gr.Row():
                        base_out = gr.Textbox(label="Base Model  ·  Before", lines=10, interactive=False)
                        ft_out   = gr.Textbox(label="LegisSum  ·  After", lines=10, interactive=False)
                        ref_out  = gr.Textbox(label="Ground Truth  ·  Reference", lines=10, interactive=False)

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

        gr.HTML('<div class="stamp">LegisSum &nbsp;·&nbsp; Mistral-7B + QLoRA &nbsp;·&nbsp; BillSum Dataset &nbsp;·&nbsp; Duke GenAI Hackathon 2026</div>')

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
