import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from model import MMBTCLIP
import os
import joblib
import numpy as np
import langchain_rag  # Custom module
from dotenv import load_dotenv

# Load environment variables if any
load_dotenv()

# --- Load Model Once ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "runs/fb_memes_vit_large/best.pt"
POLICY_INDEX = "policies/policy_index_large.pt"
LR_PATH = "results/facebook_analysis_enhanced/ensemble_lr.pkl"

print("Loading models for UI...")
# 1. MMBT
clip_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(clip_name)
clip_model = CLIPModel.from_pretrained(clip_name)
model = MMBTCLIP(clip_model, proj_dim=256, use_lora=False)

try:
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    # Peft check
    keys = list(ckpt["model"].keys())
    if any("lora" in k for k in keys):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], lora_dropout=0.05, bias="none")
        model.clip = get_peft_model(model.clip, cfg)

    model.load_state_dict(ckpt["model"], strict=False)
    print("MMBT Model loaded.")
except Exception as e:
    print(f"Error loading MMBT: {e}")

model.to(DEVICE)
model.eval()

# 2. Policies
try:
    policy_data = torch.load(POLICY_INDEX, map_location=DEVICE)
    policy_embs = F.normalize(policy_data["embeddings"].to(DEVICE), p=2, dim=-1)
    policy_texts = policy_data["texts"]
    print("Policy Index loaded.")
except Exception as e:
    print(f"Error loading Policy Index: {e}")

# 3. Dynamic Policy Gating Ensemble
PARAMS_PATH = "results/facebook_analysis_dynamic/ensemble_params.pkl"

try:
    params = joblib.load(PARAMS_PATH)
    DYN_THRESH = float(params["threshold"])
    DYN_WEIGHT = float(params["weight"]) # Boost Weight
    print(f"Dynamic Ensemble Params loaded: T={DYN_THRESH:.3f}, W={DYN_WEIGHT:.3f}")
except:
    print("Ensemble Params not found, using defaults.")
    DYN_THRESH = 0.25 # Typical CLIP Sim threshold
    DYN_WEIGHT = 0.5

# 4. Gemini Key Pool — rotate across keys on rate-limit errors
import google.generativeai as genai
import itertools, threading

GEMINI_KEYS = []
for i in range(1, 20):
    k = os.getenv(f"GEMINI_API_KEY_{i}")
    if k and k.strip():
        GEMINI_KEYS.append(k.strip())
if not GEMINI_KEYS:
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single:
        GEMINI_KEYS.append(single)
if not GEMINI_KEYS:
    raise ValueError("No GEMINI_API_KEY_* keys found in .env")

_key_cycle = itertools.cycle(GEMINI_KEYS)
_key_lock = threading.Lock()

def next_gemini_key() -> str:
    with _key_lock:
        return next(_key_cycle)

print(f"Loaded {len(GEMINI_KEYS)} Gemini API key(s).")

GEMINI_KEY = next_gemini_key()
genai.configure(api_key=GEMINI_KEY)

print("Initializing LangChain RAG Pipeline...")
try:
    langchain_rag.init_rag(
        api_keys=GEMINI_KEYS,
        policies_path="policies/example_policies.jsonl",
    )
    print("RAG Pipeline initialized.")
except Exception as e:
    print(f"RAG Init Error: {e}")

print(f"Gemini integration ready ({len(GEMINI_KEYS)} keys in rotation).")

# ── Inline SVG Icons (Lucide-style, 18px) ────────────────────────────────────
_ICONS = {
    "shield":      '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "shield-x":    '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>',
    "shield-ok":   '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>',
    "shield-warn": '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "brain":       '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M9.5 2a3.5 3.5 0 0 0-3 5.22A3.5 3.5 0 0 0 5 11a3.49 3.49 0 0 0 1.17 2.6A3.5 3.5 0 0 0 9.5 22h0a3.5 3.5 0 0 0 3.33-7.4A3.49 3.49 0 0 0 14 11a3.5 3.5 0 0 0-1.5-3.78A3.5 3.5 0 0 0 9.5 2z"/><path d="M14.5 2a3.5 3.5 0 0 1 3 5.22A3.5 3.5 0 0 1 19 11a3.49 3.49 0 0 1-1.17 2.6A3.5 3.5 0 0 1 14.5 22h0"/></svg>',
    "gauge":       '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"/><path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M20 12h2"/><path d="M2 12h2"/></svg>',
    "scale":       '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M16 2H8l-4 8 8 12 8-12-4-8z"/></svg>',
    "layers":      '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "text":        '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg>',
    "alert-tri":   '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "info":        '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "image-off":   '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><line x1="2" y1="2" x2="22" y2="22"/><path d="M10.41 10.41a2 2 0 1 1-2.83-2.83"/><line x1="13.5" y1="13.5" x2="6" y2="21"/><path d="M18 12l3 3"/><path d="M3 3h2l2 2"/><path d="M21 15V5a2 2 0 0 0-2-2H9"/></svg>',
    "x-circle":    '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
}


def _toast(icon_key, title, body, color="#a1a1aa", bg="rgba(255,255,255,0.03)", border="rgba(255,255,255,0.06)"):
    icon = _ICONS.get(icon_key, "")
    return (
        f'<div style="animation:fadeUp .35s cubic-bezier(.4,0,.2,1) both;'
        f'background:{bg};border:1px solid {border};border-radius:12px;padding:20px 24px;'
        f'max-width:480px;margin:2rem auto;">'
        f'<div style="display:flex;align-items:center;gap:10px;color:{color};margin-bottom:8px;">'
        f'{icon}<span style="font-weight:600;font-size:0.88rem;">{title}</span></div>'
        f'<p style="margin:0;color:#a1a1aa;font-size:0.82rem;line-height:1.7;padding-left:28px;">{body}</p>'
        f'</div>'
    )


def _is_blank_image(img, variance_threshold=50, edge_fraction=0.005):
    """Detect blank/near-uniform images (solid color, white, black, etc.)."""
    arr = np.array(img.convert("RGB").resize((128, 128)))
    if arr.std() < variance_threshold:
        return True
    gray = np.mean(arr, axis=2).astype(np.uint8)
    sx = np.abs(np.diff(gray, axis=1)).sum()
    sy = np.abs(np.diff(gray, axis=0)).sum()
    total_edge_energy = (sx + sy) / gray.size
    return total_edge_energy < edge_fraction


def predict(image, text, progress=gr.Progress()):
    ocr_source = ""
    if image is None:
        if not text or text.strip() == "":
            return _toast(
                "info", "No Input Provided",
                "Please upload an image or enter a caption to start the analysis.",
                color="#818cf8", bg="rgba(99,102,241,0.06)", border="rgba(99,102,241,0.12)",
            )
        image = Image.new('RGB', (224, 224), color='white')
        ocr_source = "(Text-only analysis: No image provided)"
        skip_ocr = True
    else:
        if _is_blank_image(image):
            if not text or text.strip() == "":
                return _toast(
                    "image-off", "Blank Image Detected",
                    "The uploaded image appears to be blank or a solid color. "
                    "Please upload a meme with visible content, or provide a caption for text-only analysis.",
                    color="#f59e0b", bg="rgba(245,158,11,0.06)", border="rgba(245,158,11,0.12)",
                )
            image = Image.new('RGB', (224, 224), color='white')
            ocr_source = "(Blank image detected — using text-only analysis)"
            skip_ocr = True
        else:
            skip_ocr = False

    progress(0.1, desc="Initializing detection pipeline...")
    
    # Ensure text fallback
    if not text: text = ""
    
    # OCR Fallback (Gemini) — rotate keys on rate-limit errors
    if not skip_ocr and text.strip() == "":
        progress(0.2, desc="No caption found — running Gemini OCR...")
        print("No text provided. Running Gemini OCR...")
        success = False
        last_error = ""
        ocr_model = "gemini-flash-latest"

        for attempt in range(len(GEMINI_KEYS)):
            key = next_gemini_key()
            print(f"[OCR] Attempt {attempt+1}/{len(GEMINI_KEYS)} with key ...{key[-6:]}")
            try:
                genai.configure(api_key=key)
                gemini_model = genai.GenerativeModel(ocr_model)
                response = gemini_model.generate_content(
                    ["Extract all text from this image exactly as it appears.", image]
                )
                text = response.text.strip()
                ocr_source = f"(Extracted via {ocr_model})"
                print(f"[OCR] Success with key ...{key[-6:]}")
                success = True
                break
            except Exception as e:
                err_str = str(e).lower()
                print(f"[OCR] Failed key ...{key[-6:]}: {e}")
                last_error = str(e)
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    continue
                break

        if not success:
            text = " "
            ocr_source = f"(OCR failed after {len(GEMINI_KEYS)} keys. Last: {last_error[:50]}...)"
        
    if not text: text = " " # Fallback if everything else fails
    
    progress(0.4, desc="Fusing multimodal features...")
    # Prepare Input
    enc = processor(text=[text], images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    
    # MMBT Inference
    try:
        with torch.no_grad():
            pixel_values = enc["pixel_values"].to(DEVICE)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            
            logits = model(input_ids, attention_mask, pixel_values)
            prob_base = float(torch.sigmoid(logits).item())
            
            progress(0.6, desc="Querying policy knowledge base...")
            # Policy Inference
            base_clip = model.clip.base_model.model if hasattr(model.clip, "base_model") else model.clip
            
            img_feats = F.normalize(base_clip.get_image_features(pixel_values=pixel_values), p=2, dim=-1)
            txt_feats = F.normalize(base_clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask), p=2, dim=-1)
            
            sim_img = torch.matmul(img_feats, policy_embs.T)
            sim_txt = torch.matmul(txt_feats, policy_embs.T)
            
            score_img, idx_img = sim_img.max(dim=1)
            score_txt, idx_txt = sim_txt.max(dim=1)
            
            prob_policy = max(score_img.item(), score_txt.item())
            prob_policy_norm = (prob_policy + 1) / 2.0
            
            # Dynamic Gating Logic
            is_strong_match = prob_policy > DYN_THRESH
            
            if is_strong_match:
                # Boost Policy Influence
                prob_final = (1 - DYN_WEIGHT) * prob_base + DYN_WEIGHT * prob_policy_norm
                source = f"Policy Boost (Match > {DYN_THRESH:.2f})"
            else:
                # Rely on Base Model
                prob_final = prob_base
                source = "Base Model (No Policy Match)"
                
            # Get Top Policy
            if score_img > score_txt:
                top_policy = policy_texts[idx_img]
                match_source = "Image Match"
            else:
                top_policy = policy_texts[idx_txt]
                match_source = "Text Match"

        # --- RAG Reasoner ---
        initial_label = "HATEFUL" if prob_final > 0.5 else "NON-HATEFUL"
        
        progress(0.8, desc="Running RAG reasoner...")
        print(f"Calling RAG Reasoner for: {text[:50]}...")
        reasoning, matched_policies, final_label = langchain_rag.get_rag_explanation(text, initial_label, prob_final)
        
        progress(1.0, desc="Analysis complete")
        
        # --- Result Formatting with Iconography ---
        label_cfg = {
            "HATEFUL":       ("shield-x",   "#ef4444", "rgba(239,68,68,0.08)", "rgba(239,68,68,0.18)", "#fca5a5"),
            "INAPPROPRIATE": ("shield-warn", "#f59e0b", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.18)", "#fcd34d"),
            "NON-HATEFUL":   ("shield-ok",   "#10b981", "rgba(16,185,129,0.08)", "rgba(16,185,129,0.18)", "#6ee7b7"),
        }
        icon_key, badge_color, callout_bg, callout_border, callout_fg = label_cfg.get(
            final_label, ("shield", "#71717a", "rgba(113,113,122,0.08)", "rgba(113,113,122,0.18)", "#a1a1aa")
        )
        badge_icon = _ICONS[icon_key].replace('stroke="currentColor"', f'stroke="#fff"')

        confidence_pct = f"{prob_final:.1%}"
        text_preview = f"{text[:180]}{'...' if len(text) > 180 else ''}"

        policy_heading = "Policy Violations" if final_label in ("HATEFUL", "INAPPROPRIATE") else "Policy Review"
        policy_list = "\n".join([f"- {p}" for p in matched_policies]) if matched_policies else "_No policy violations detected._"

        mmbt_flag = "Flagged" if prob_base > 0.5 else "Clear"
        policy_flag = "Active" if is_strong_match else "Inactive"

        icon_brain  = _ICONS["brain"].replace('stroke="currentColor"', f'stroke="{callout_fg}"')
        icon_gauge  = _ICONS["gauge"].replace('stroke="currentColor"', 'stroke="#818cf8"')
        icon_text   = _ICONS["text"].replace('stroke="currentColor"', 'stroke="#818cf8"')
        icon_scale  = _ICONS["scale"].replace('stroke="currentColor"', 'stroke="#818cf8"')
        icon_layers = _ICONS["layers"].replace('stroke="currentColor"', 'stroke="#818cf8"')

        result_md = f"""<div style="font-family:'Inter',system-ui,sans-serif;animation:fadeUp .4s cubic-bezier(.4,0,.2,1) both;">

<div style="display:inline-flex;align-items:center;gap:8px;background:{badge_color};color:#fff;padding:8px 16px;border-radius:8px;font-weight:700;font-size:0.76rem;letter-spacing:0.06em;text-transform:uppercase;">{badge_icon} {final_label}</div>

<div style="margin-top:18px;padding:16px 20px;background:{callout_bg};border:1px solid {callout_border};border-left:3px solid {badge_color};border-radius:0 10px 10px 0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;color:{callout_fg};font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">{icon_brain} Reasoner Insights</div>
<p style="margin:0;color:{callout_fg};font-size:0.84rem;line-height:1.75;">{reasoning}</p>
</div>

---

<div style="display:flex;flex-wrap:wrap;gap:20px;align-items:center;">
<div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#a1a1aa;">{icon_gauge} <strong style="color:#fafafa;">Confidence</strong>&nbsp; <code style="background:#18181f;border:1px solid rgba(255,255,255,0.06);padding:2px 8px;border-radius:5px;font-size:0.78rem;color:#818cf8;">{confidence_pct}</code></div>
<div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#a1a1aa;">{icon_layers} <strong style="color:#fafafa;">Attribution</strong>&nbsp; <code style="background:#18181f;border:1px solid rgba(255,255,255,0.06);padding:2px 8px;border-radius:5px;font-size:0.78rem;color:#818cf8;">{source}</code></div>
</div>

<div style="display:flex;align-items:flex-start;gap:6px;margin-top:12px;font-size:0.82rem;color:#a1a1aa;">{icon_text} <span><strong style="color:#fafafa;">Analyzed Text</strong>&nbsp; {text_preview} <span style="color:#52525b;font-size:0.72rem;">{ocr_source}</span></span></div>

---

<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;color:#818cf8;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">{icon_scale} {policy_heading}</div>

{policy_list}

---

| Component | Value | Status |
| :--- | :--- | :--- |
| MMBT Base Score | `{prob_base:.4f}` | {mmbt_flag} |
| Policy Similarity | `{prob_policy_norm:.4f}` | {policy_flag} |
| Raw CLIP Match | `{prob_policy:.4f}` | Threshold: {DYN_THRESH} |

</div>"""
        return result_md
    except Exception as e:
        return _toast(
            "x-circle", "Inference Error",
            f"Something went wrong during analysis: {str(e)[:200]}",
            color="#ef4444", bg="rgba(239,68,68,0.06)", border="rgba(239,68,68,0.12)",
        )

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Keyframes ──────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
    50%      { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0.08); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes progressPulse {
    0%   { opacity: 0.4; }
    50%  { opacity: 1; }
    100% { opacity: 0.4; }
}

:root {
    --bg-base: #09090b;
    --bg-raised: #121218;
    --bg-surface: #18181f;
    --bg-overlay: #1e1e27;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-default: rgba(255, 255, 255, 0.09);
    --border-focus: rgba(99, 102, 241, 0.5);
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --accent-muted: rgba(99, 102, 241, 0.12);
    --accent-glow: rgba(99, 102, 241, 0.2);
    --green: #10b981;
    --green-muted: rgba(16, 185, 129, 0.1);
    --red: #ef4444;
    --red-muted: rgba(239, 68, 68, 0.1);
    --amber: #f59e0b;
    --amber-muted: rgba(245, 158, 11, 0.1);
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --ease: cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html {
    scroll-behavior: smooth !important;
}

.gradio-container {
    background: var(--bg-base) !important;
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    min-height: 100vh !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

footer { display: none !important; }

/* ── Smooth Scrollbar ───────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }
* { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent; }

/* ── Layout ─────────────────────────────────── */
.app-wrapper {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 0 2rem !important;
}

/* ── Header ─────────────────────────────────── */
.header-bar {
    padding: 2.5rem 0 2rem !important;
    margin-bottom: 1.5rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    background: transparent !important;
    animation: fadeIn 0.6s var(--ease) both !important;
}

.brand-icon {
    text-align: center !important;
    margin-bottom: 10px !important;
}
.brand-icon .icon-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--accent-muted);
    border: 1px solid rgba(99, 102, 241, 0.15);
}
.brand-name h1 {
    font-family: var(--font-sans) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.04em !important;
    margin: 0 !important;
    text-align: center !important;
}

.brand-tagline p {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    text-align: center !important;
    margin: 0.4rem 0 0 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

.header-chips {
    margin-top: 1.1rem !important;
    text-align: center !important;
}
.header-chips .chip-row {
    display: inline-flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
}
.header-chips .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-sans);
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    padding: 5px 12px;
    border-radius: 100px;
    letter-spacing: 0.02em;
    transition: border-color 0.2s var(--ease), background 0.2s var(--ease);
}
.header-chips .chip:hover {
    border-color: var(--border-default);
    background: var(--bg-overlay);
}
.header-chips .chip svg {
    opacity: 0.5;
    flex-shrink: 0;
}

/* ── Cards ──────────────────────────────────── */
.input-card {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    animation: slideInLeft 0.5s var(--ease) 0.15s both !important;
    transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease) !important;
}
.input-card:hover {
    border-color: rgba(255,255,255,0.12) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
}

.card-label {
    margin-bottom: 1rem !important;
}
.card-label .label-inner {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-sans);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.card-label .label-inner svg {
    opacity: 0.5;
}
.card-label p {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    margin: 0 !important;
}

/* ── Inputs ─────────────────────────────────── */
.gradio-container textarea,
.gradio-container input[type="text"] {
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: all 0.25s var(--ease) !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-muted) !important;
    outline: none !important;
}

.gradio-container label {
    font-family: var(--font-sans) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
}

.gradio-container .image-container,
.gradio-container .upload-container {
    background: var(--bg-surface) !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color 0.25s var(--ease) !important;
}
.gradio-container .upload-container:hover {
    border-color: var(--accent) !important;
}

/* ── Button ─────────────────────────────────── */
.run-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem 1.5rem !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #fff !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    transition: all 0.25s var(--ease) !important;
    position: relative !important;
    overflow: hidden !important;
}
.run-btn:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}
.run-btn:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ── Features row ───────────────────────────── */
.features-row {
    margin-top: 1rem !important;
    padding: 0.85rem 1rem !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}
.features-row p {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    line-height: 1.8 !important;
    margin: 0 !important;
    text-align: center !important;
}
.features-row strong {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

/* ── Results panel ──────────────────────────── */
.results-card {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    min-height: 500px !important;
    animation: slideInRight 0.5s var(--ease) 0.2s both !important;
    transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease) !important;
}
.results-card:hover {
    border-color: rgba(255,255,255,0.12) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
}

/* Skeleton loading state — shown while Gradio streams */
.results-card .generating {
    position: relative !important;
}
.results-card .generating::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    z-index: 5 !important;
    border-radius: var(--radius-lg) !important;
    pointer-events: none !important;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(99, 102, 241, 0.04) 40%,
        rgba(99, 102, 241, 0.08) 50%,
        rgba(99, 102, 241, 0.04) 60%,
        transparent 100%
    ) !important;
    background-size: 800px 100% !important;
    animation: shimmer 1.8s infinite linear !important;
}

/* Skeleton placeholder lines */
.skeleton-wrap {
    padding: 0.5rem 0 !important;
    animation: fadeIn 0.3s var(--ease) both !important;
}

/* Progress bar override */
.gradio-container .progress-bar {
    background: var(--accent) !important;
    border-radius: 100px !important;
    transition: width 0.4s var(--ease) !important;
}
.gradio-container .progress-bar-wrap {
    background: var(--bg-surface) !important;
    border-radius: 100px !important;
    height: 4px !important;
    overflow: hidden !important;
}
.gradio-container .progress-text {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    animation: progressPulse 1.5s ease-in-out infinite !important;
}

/* ── Result content styling ─────────────────── */
.results-card .prose,
.results-card .md,
.results-card .markdown {
    animation: fadeUp 0.4s var(--ease) both !important;
}

.results-card .prose h1, .results-card .prose h2, .results-card .prose h3,
.results-card .md h1, .results-card .md h2, .results-card .md h3 {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.results-card .prose p, .results-card .prose li,
.results-card .md p, .results-card .md li {
    font-family: var(--font-sans) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    line-height: 1.75 !important;
}

.results-card .prose code, .results-card .md code {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    color: var(--accent-hover) !important;
}

.results-card .prose table, .results-card .md table {
    width: 100% !important;
    font-size: 0.8rem !important;
    border-collapse: collapse !important;
    margin: 0.5rem 0 !important;
}
.results-card .prose th, .results-card .md th {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border-default) !important;
    text-align: left !important;
    background: var(--bg-surface) !important;
}
.results-card .prose th:first-child, .results-card .md th:first-child {
    border-radius: var(--radius-sm) 0 0 0 !important;
}
.results-card .prose th:last-child, .results-card .md th:last-child {
    border-radius: 0 var(--radius-sm) 0 0 !important;
}
.results-card .prose td, .results-card .md td {
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    transition: background 0.15s ease !important;
}
.results-card .prose tr:hover td, .results-card .md tr:hover td {
    background: rgba(255,255,255,0.02) !important;
}

.results-card .prose hr, .results-card .md hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.25rem 0 !important;
}

.results-card .prose strong, .results-card .md strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Idle placeholder ───────────────────────── */
.idle-state p {
    text-align: center !important;
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    padding: 4rem 2rem !important;
    line-height: 1.8 !important;
}
.idle-state p em {
    color: var(--text-muted) !important;
}

/* ── Footer ─────────────────────────────────── */
.app-footer {
    border-top: 1px solid var(--border-subtle) !important;
    margin-top: 2.5rem !important;
    padding: 1.5rem 0 2rem !important;
    animation: fadeIn 0.6s var(--ease) 0.4s both !important;
}
.app-footer p {
    text-align: center !important;
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    line-height: 2 !important;
    margin: 0 !important;
    letter-spacing: 0.02em !important;
}
.app-footer strong {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}
.app-footer .footer-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}
.app-footer .footer-inner svg {
    opacity: 0.3;
}

/* ── Global transitions on Gradio internals ── */
.gradio-container .block {
    transition: opacity 0.3s var(--ease), transform 0.3s var(--ease) !important;
}
.gradio-container .block.hidden {
    opacity: 0 !important;
    transform: translateY(8px) !important;
}

/* ── Progress descriptions ──────────────────── */
.gradio-container .progress-text span,
.gradio-container .progress-level span {
    font-family: var(--font-sans) !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.02em !important;
}

/* ── Toast animation helper ─────────────────── */
.results-card .prose > div:first-child,
.results-card .md > div:first-child {
    animation: fadeUp 0.35s var(--ease) both !important;
}
"""

SKELETON_HTML = """<div class="skeleton-wrap">
<div style="display:flex;flex-direction:column;gap:12px;padding:3rem 1rem;">
<div style="width:90px;height:28px;border-radius:6px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
<div style="width:100%;height:14px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;margin-top:8px;"></div>
<div style="width:85%;height:14px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
<div style="width:60%;height:14px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
<div style="width:100%;height:1px;background:rgba(255,255,255,0.06);margin:12px 0;"></div>
<div style="display:flex;gap:16px;">
<div style="width:120px;height:12px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
<div style="width:100px;height:12px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
</div>
<div style="width:100%;height:1px;background:rgba(255,255,255,0.06);margin:12px 0;"></div>
<div style="width:70%;height:12px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
<div style="width:50%;height:12px;border-radius:4px;background:linear-gradient(90deg,rgba(255,255,255,0.04) 25%,rgba(255,255,255,0.08) 50%,rgba(255,255,255,0.04) 75%);background-size:400px 100%;animation:shimmer 1.8s infinite linear;"></div>
</div>
</div>"""

IDLE_MSG = """<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;gap:16px;animation:fadeIn .5s cubic-bezier(.4,0,.2,1) both;">
<div style="width:56px;height:56px;border-radius:14px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.12);display:flex;align-items:center;justify-content:center;">
<svg width="26" height="26" fill="none" viewBox="0 0 24 24" stroke="#6366f1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
</div>
<p style="color:#71717a;font-size:0.84rem;text-align:center;margin:0;line-height:1.8;max-width:280px;">Upload an image or enter a caption to start content analysis.</p>
<div style="display:flex;gap:6px;margin-top:4px;">
<span style="font-size:0.65rem;color:#52525b;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:3px 8px;border-radius:100px;">Image + Text</span>
<span style="font-size:0.65rem;color:#52525b;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:3px 8px;border-radius:100px;">Text Only</span>
<span style="font-size:0.65rem;color:#52525b;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:3px 8px;border-radius:100px;">Image + OCR</span>
</div>
</div>"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.zinc,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#09090b",
    body_background_fill_dark="#09090b",
    block_background_fill="#121218",
    block_background_fill_dark="#121218",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.09)",
    block_radius="12px",
    input_background_fill="#18181f",
    input_background_fill_dark="#18181f",
    input_border_color="rgba(255,255,255,0.06)",
    input_border_width="1px",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_hover="#818cf8",
    button_primary_text_color="#ffffff",
    body_text_color="#fafafa",
    body_text_color_dark="#fafafa",
    body_text_color_subdued="#a1a1aa",
)


def show_skeleton():
    return SKELETON_HTML


with gr.Blocks(css=CSS, theme=THEME, title="Discri-Net") as demo:
    with gr.Column(elem_classes=["app-wrapper"]):

        # ── Header ────────────────────────────────
        with gr.Column(elem_classes=["header-bar"]):
            gr.HTML(
                '<div class="brand-icon">'
                '<div class="icon-box">'
                '<svg width="22" height="22" fill="none" viewBox="0 0 24 24" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
                '<polyline points="9 12 11 14 15 10"/>'
                '</svg>'
                '</div>'
                '</div>'
            )
            gr.Markdown("# Discri-Net", elem_classes=["brand-name"])
            gr.Markdown("Multimodal Hate Speech Detection & RAG Reasoning", elem_classes=["brand-tagline"])
            gr.HTML(
                '<div class="header-chips"><div class="chip-row">'
                '<span class="chip"><svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg> CLIP ViT-L/14</span>'
                '<span class="chip"><svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg> MMBT Fusion</span>'
                '<span class="chip"><svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> FAISS Retrieval</span>'
                '<span class="chip"><svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2a3.5 3.5 0 0 0-3 5.22A3.5 3.5 0 0 0 5 11a3.49 3.49 0 0 0 1.17 2.6A3.5 3.5 0 0 0 9.5 22h0a3.5 3.5 0 0 0 3.33-7.4A3.49 3.49 0 0 0 14 11a3.5 3.5 0 0 0-1.5-3.78A3.5 3.5 0 0 0 9.5 2z"/><path d="M14.5 2a3.5 3.5 0 0 1 3 5.22A3.5 3.5 0 0 1 19 11a3.49 3.49 0 0 1-1.17 2.6A3.5 3.5 0 0 1 14.5 22h0"/></svg> Gemini RAG</span>'
                '<span class="chip"><svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> 50+ Policies</span>'
                '</div></div>'
            )

        # ── Main Layout ───────────────────────────
        with gr.Row(equal_height=False):

            with gr.Column(scale=5, min_width=380):
                with gr.Column(elem_classes=["input-card"]):
                    gr.HTML(
                        '<div class="card-label"><div class="label-inner">'
                        '<svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
                        '<polyline points="17 8 12 3 7 8"/>'
                        '<line x1="12" y1="3" x2="12" y2="15"/>'
                        '</svg> INPUT</div></div>'
                    )

                    input_img = gr.Image(
                        type="pil",
                        label="Image",
                        interactive=True,
                        height=250,
                    )

                    input_text = gr.Textbox(
                        label="Caption / Meme Text",
                        placeholder="Enter text, or leave empty for automatic OCR...",
                        lines=3,
                    )

                    btn = gr.Button(
                        "Analyze",
                        variant="primary",
                        elem_classes=["run-btn"],
                    )

                    with gr.Column(elem_classes=["features-row"]):
                        gr.Markdown(
                            "**Multimodal MMBT** fusion &middot; "
                            "**FAISS** policy retrieval &middot; "
                            "**Gemini** OCR + RAG"
                        )

            with gr.Column(scale=7, min_width=460):
                with gr.Column(elem_classes=["results-card"]):
                    gr.HTML(
                        '<div class="card-label"><div class="label-inner">'
                        '<svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                        '<polyline points="14 2 14 8 20 8"/>'
                        '<line x1="16" y1="13" x2="8" y2="13"/>'
                        '<line x1="16" y1="17" x2="8" y2="17"/>'
                        '<polyline points="10 9 9 9 8 9"/>'
                        '</svg> RESULTS</div></div>'
                    )
                    output_md = gr.Markdown(
                        IDLE_MSG,
                        elem_id="results-display",
                        elem_classes=["idle-state"],
                    )

        # ── Footer ────────────────────────────────
        with gr.Column(elem_classes=["app-footer"]):
            gr.HTML(
                '<div class="footer-inner">'
                '<svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="#71717a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>'
                '</svg>'
                '<p style="margin:0;"><strong>Discri-Net</strong> &mdash; CLIP ViT-Large &middot; Gemini 2.0 Flash &middot; LangChain FAISS</p>'
                '<p style="margin:0;">Built by &nbsp;<strong>22K-4104</strong> &nbsp;&middot;&nbsp; <strong>22K-4105</strong> &nbsp;&middot;&nbsp; <strong>22K-8707</strong></p>'
                '</div>'
            )

        btn.click(fn=show_skeleton, inputs=None, outputs=output_md).then(
            fn=predict,
            inputs=[input_img, input_text],
            outputs=output_md,
            api_name="analyze",
        )

if __name__ == "__main__":
    demo.launch(share=True)
