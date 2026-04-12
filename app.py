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
    model.to(DEVICE)
    model.eval()
    print("MMBT Model loaded.")
except Exception as e:
    print(f"Error loading MMBT: {e}")

# 2. Policy
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

# 4. OCR & RAG Reasoner (Gemini)
import google.generativeai as genai

# Use environment variable or fallback to provided key
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAk-rQIMbww_LBois-ewBhriaDpWgnmsA0")
genai.configure(api_key=GEMINI_KEY)

print("Initializing LangChain RAG Pipeline...")
try:
    langchain_rag.init_rag(api_key=GEMINI_KEY, policies_path="policies/example_policies.jsonl")
    print("RAG Pipeline initialized.")
except Exception as e:
    print(f"RAG Init Error: {e}")

print("Initialized Gemini OCR & Reasoner Integration.")

def predict(image, text, progress=gr.Progress()):
    if image is None: return "### ⚠️ Please upload an image to begin."
    
    progress(0.1, desc="🚀 Initializing detection pipeline...")
    
    # OCR Fallback (Gemini)
    ocr_source = ""
    # Ensure text fallback
    if not text: text = ""
    
    # OCR Fallback (Gemini)
    ocr_source = ""
    # Ensure text fallback
    if not text: text = ""
    
    # List of models to try (fallback strategy for Free Tier)
    GEMINI_MODELS = [
        "gemini-flash-latest",            # Alias to stable flash
    ]

    if text.strip() == "":
        progress(0.2, desc="🔍 No caption found. Running Gemini OCR...")
        print("No text provided. Running Gemini OCR...")
        success = False
        last_error = ""
        
        for model_name in GEMINI_MODELS:
            print(f"Trying Gemini Model: {model_name}...")
            try:
                gemini_model = genai.GenerativeModel(model_name)
                # Gemini accepts PIL Image directly
                response = gemini_model.generate_content(["Extract all text from this image exactly as it appears.", image])
                text = response.text.strip()
                ocr_source = f"(Extracted via {model_name})"
                print(f"Gemini Extraction Success: {text}")
                success = True
                break # Stop if successful
            except Exception as e:
                print(f"Failed with {model_name}: {e}")
                last_error = str(e)
                # Continue to next model
        
        if not success:
             text = " "
             ocr_source = f"(OCR Failed: All models exhausted. Last error: {last_error[:50]}...)"
        
    if not text: text = " " # Fallback if OCR fails
    
    progress(0.4, desc="🧬 Fusing multimodal CLIP features...")
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
            
            progress(0.6, desc="⚖️ Querying policy knowledge base...")
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
        label = "HATEFUL" if prob_final > 0.5 else "NON-HATEFUL"
        
        progress(0.8, desc="🧠 Consultative Gemini Reasoner active...")
        print(f"Calling RAG Reasoner for: {text[:50]}...")
        reasoning, matched_policies = langchain_rag.get_rag_explanation(text, label, prob_final)
        
        progress(1.0, desc="✅ Analysis complete!")
        
        # --- Rich Markdown Formatting ---
        badge_color = "#ff4b4b" if label == "HATEFUL" else "#00c853"
        badge_html = f'<div style="background-color: {badge_color}; color: white; padding: 12px 24px; border-radius: 12px; display: inline-block; font-weight: 800; font-size: 22px; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">{label}</div>'
        
        policy_section_title = "⚖️ Policy Violations" if label == "HATEFUL" else "⚖️ Contextual Policy Review"
        policy_list = "\n".join([f"- **{p.split('.')[0]}** - {'.'.join(p.split('.')[1:]).strip()}" for p in matched_policies])
        
        result_md = f"""
{badge_html}

### 🧠 Reasoner Insights
{reasoning}

---

### 📊 Detection Metadata
- **Analysis Confidence**: `{prob_final:.2%}`
- **Primary Attribution**: `{source}`
- **Text Analysis**: `{text[:200]}{'...' if len(text) > 200 else ''}` {ocr_source}

### {policy_section_title}
{policy_list if matched_policies else "_No specific policies retrieved for this content (Confidence too low)._"}

---

### 🔬 Technical Breakdown
| Component | Metric | Value | Status |
| :--- | :--- | :--- | :--- |
| **Multimodal MMBT** | Base Probability | {prob_base:.4f} | {"⚠️ Flagged" if prob_base > 0.5 else "✅ Safe"} |
| **Policy Stacking** | Normalized Similarity | {prob_policy_norm:.4f} | {"🔥 Boosting" if is_strong_match else "ℹ️ Passive"} |
| **Raw Match** | CLIP Similarity | {prob_policy:.4f} | (Threshold: {DYN_THRESH}) |
"""
        return result_md
    except Exception as e:
        return f"### ❌ Error during inference\n{e}"

# --- CUSTOM CSS ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');

:root {
    --primary: #ff4b4b;
    --secondary: #2d3436;
    --font-main: 'Inter', sans-serif;
    --font-heading: 'Outfit', sans-serif;
}

body { font-family: var(--font-main); }

.container { 
    max-width: 1300px; 
    margin: auto; 
    padding: 2rem 1rem; 
}
.glass-panel {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-panel:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px 0 rgba(0, 0, 0, 0.55);
}
.title-text {
    font-family: var(--font-heading);
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ff4b4b 0%, #ff8a8a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem !important;
    letter-spacing: -1.5px;
}
.header-box {
    text-align: center;
    margin-bottom: 4rem;
}
.analyze-btn {
    background: linear-gradient(135deg, #ff4b4b 0%, #c0392b 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    height: 3.5rem !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
    transition: all 0.2s ease !important;
}
.analyze-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4) !important;
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes=["container"]):
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown("# 🛡️ Discri-Net: Hate Speech Guard", elem_classes=["title-text"])
            gr.Markdown("### Multimodal Meme Analysis with Gemini RAG Reasoning")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Column(elem_classes=["glass-panel"]):
                    input_img = gr.Image(type="pil", label="Meme Image", interactive=True)
                    input_text = gr.Textbox(
                        label="⌨️ Meme Text (Caption)", 
                        placeholder="Leave empty for Gemini OCR extraction...",
                        lines=3
                    )
                    btn = gr.Button("🚀 Analyze Meme", variant="primary", size="lg", elem_classes=["analyze-btn"])
                    
                    gr.Markdown("#### 💡 Pro Tips")
                    gr.Markdown("""
                    - **Precision**: Higher resolution = Better OCR accuracy.
                    - **Logic**: The RAG Reasoner uses NLI to maps memes to official safety policies.
                    - **Speed**: Built on CLIP-VIT-Large for production-grade feature extraction.
                    """)

            with gr.Column(scale=3):
                with gr.Column(elem_classes=["glass-panel"]):
                    output_md = gr.Markdown("Analysis results will appear here after clicking Analyze.")
                    
        btn.click(fn=predict, inputs=[input_img, input_text], outputs=output_md)

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="slate"),
        css=CSS
    )
