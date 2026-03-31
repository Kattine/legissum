"""
app.py — LegisSum FastAPI backend + HTML frontend
──────────────────────────────────────────────────
Run:
    python app.py
Opens at http://localhost:8000
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
ADAPTER_PATH = ROOT / "models" / "billsum-lora"
OUTPUTS_DIR  = ROOT / "outputs"
STATIC_DIR   = ROOT / "static"

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
HF_ADAPTER_ID = os.environ.get("HF_MODEL_ID", "")

SYSTEM_INSTRUCTION = (
    "You are a legislative assistant helping everyday citizens understand U.S. law. "
    "Read the following Congressional bill and write a clear, concise summary. "
    "Focus on: what the bill does, who it affects, and its key provisions. "
    "Use plain English — no legal jargon."
)

# ── Model state ────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_label = ""
_model_ready = False
_model_error = None


def _load_model_bg():
    global _model, _tokenizer, _model_label, _model_ready, _model_error
    try:
        print(f"\n⬇  Loading {BASE_MODEL_ID}…")
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

        if PEFT_AVAILABLE and ADAPTER_PATH.exists():
            _model = PeftModel.from_pretrained(_model, str(ADAPTER_PATH))
            _model_label = "Fine-tuned LegisSum (local adapter)"
            print(f"✅ Loaded LoRA adapter from {ADAPTER_PATH}")
        elif PEFT_AVAILABLE and HF_ADAPTER_ID:
            _model = PeftModel.from_pretrained(_model, HF_ADAPTER_ID)
            _model_label = f"Fine-tuned LegisSum ({HF_ADAPTER_ID})"
            print(f"✅ Loaded adapter from HF Hub: {HF_ADAPTER_ID}")
        else:
            _model_label = "Base model (no adapter found)"
            print("⚠  No adapter — running base model only")

        _model_ready = True
        print("✅ Model ready\n")
    except Exception as e:
        _model_error = str(e)
        print(f"❌ Model load failed: {e}")


# Start loading in background so the server starts immediately
threading.Thread(target=_load_model_bg, daemon=True).start()


def _generate(bill_text: str, max_new_tokens: int = 300) -> str:
    prompt = (
        f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Bill:\n{bill_text[:3000]}\n\n"
        f"### Summary:\n"
    )
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True).to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Summary:" in decoded:
        return decoded.split("### Summary:")[-1].strip()
    return decoded.strip()


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="LegisSum")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def status():
    if _model_error:
        return {"status": "error", "message": _model_error}
    if _model_ready:
        return {"status": "ready", "label": _model_label}
    return {"status": "loading"}


class SummarizeRequest(BaseModel):
    text: str


@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    if not _model_ready:
        return JSONResponse({"error": "Model still loading, please wait"}, status_code=503)
    summary = _generate(req.text)
    return {"summary": summary, "model_label": _model_label}


@app.get("/api/comparisons")
def comparisons():
    path = OUTPUTS_DIR / "sample_comparisons.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


@app.get("/api/rouge")
def rouge():
    path = OUTPUTS_DIR / "rouge_scores.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


if __name__ == "__main__":
    print("\n🏛  LegisSum — starting at http://localhost:8080\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
