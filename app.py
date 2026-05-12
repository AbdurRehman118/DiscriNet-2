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

# Use environment variable
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please ensure it is set in your .env file.")
genai.configure(api_key=GEMINI_KEY)

print("Initializing LangChain RAG Pipeline...")
try:
    langchain_rag.init_rag(api_key=GEMINI_KEY, policies_path="policies/example_policies.jsonl")
    print("RAG Pipeline initialized.")
except Exception as e:
    print(f"RAG Init Error: {e}")

print("Initialized Gemini OCR & Reasoner Integration.")

def predict(image, text, progress=gr.Progress()):
    ocr_source = "" # Initialize to avoid UnboundLocalError
    if image is None:
        if not text or text.strip() == "":
            return "### ⚠️ Please provide at least a caption or an image to begin."
        # Create a blank white image if no image is provided but text is (placeholder for multimodal model)
        image = Image.new('RGB', (224, 224), color='white')
        ocr_source = "(Text-only analysis: No image provided)"
        skip_ocr = True
    else:
        skip_ocr = False

    progress(0.1, desc="🚀 Initializing detection pipeline...")
    
    # Ensure text fallback
    if not text: text = ""
    
    # OCR Fallback (Gemini) - Only if an image exists and text is empty
    if not skip_ocr and text.strip() == "":
        progress(0.2, desc="🔍 No caption found. Running Gemini OCR...")
        print("No text provided. Running Gemini OCR...")
        success = False
        last_error = ""
        
        # List of models to try (fallback strategy for Free Tier)
        GEMINI_MODELS = ["gemini-flash-latest"]

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
        
        if not success:
             text = " "
             ocr_source = f"(OCR Failed: All models exhausted. Last error: {last_error[:50]}...)"
        
    if not text: text = " " # Fallback if everything else fails
    
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
        initial_label = "HATEFUL" if prob_final > 0.5 else "NON-HATEFUL"
        
        progress(0.8, desc="🧠 Consultative Gemini Reasoner active...")
        print(f"Calling RAG Reasoner for: {text[:50]}...")
        reasoning, matched_policies, final_label = langchain_rag.get_rag_explanation(text, initial_label, prob_final)
        
        progress(1.0, desc="✅ Analysis complete!")
        
        # --- Rich Markdown Formatting ---
        if final_label == "HATEFUL":
            badge_color = "#ff4b4b" # Red
        elif final_label == "INAPPROPRIATE":
            badge_color = "#ffa500" # Orange
        else:
            badge_color = "#00c853" # Green
            
        badge_html = f'<div style="background-color: {badge_color}; color: white; padding: 12px 24px; border-radius: 12px; display: inline-block; font-weight: 800; font-size: 22px; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">{final_label}</div>'
        
        policy_section_title = "⚖️ Policy Violations" if final_label == "HATEFUL" or final_label == "INAPPROPRIATE" else "⚖️ Contextual Policy Review"
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

# --- PREMIUM CUSTOM CSS ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --primary: #FF3CAC;
    --primary-gradient: linear-gradient(225deg, #FF3CAC 0%, #784BA0 50%, #2B86C5 100%);
    --bg-dark: #0f172a;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --font-heading: 'Outfit', sans-serif;
    --font-body: 'Inter', sans-serif;
}

.gradio-container {
    background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%) !important;
    color: #f8fafc !important;
}

.container { 
    max-width: 1200px !important; 
    margin: auto !important; 
    padding: 3rem 1.5rem !important; 
}

.title-text {
    font-family: var(--font-heading) !important;
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    background: linear-gradient(135deg, #fff 30%, #94a3b8 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.04em !important;
    margin-bottom: 0.5rem !important;
}

.subtitle-text {
    font-family: var(--font-body) !important;
    font-size: 1.1rem !important;
    text-align: center !important;
    color: #94a3b8 !important;
    margin-bottom: 3.5rem !important;
    font-weight: 400 !important;
}

.glass-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    padding: 2rem !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
}

.input-box {
    border: 1px solid var(--glass-border) !important;
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 12px !important;
}

.analyze-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem !important;
    font-family: var(--font-heading) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    color: white !important;
    box-shadow: 0 10px 20px -5px rgba(255, 60, 172, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

.analyze-btn:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 15px 30px -5px rgba(255, 60, 172, 0.6) !important;
}

.analyze-btn:active {
    transform: translateY(0) !important;
}

.pro-tip-card {
    background: rgba(59, 130, 246, 0.05) !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin-top: 2rem !important;
}

.pro-tip-title {
    color: #60a5fa !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* Custom Result Card Styling */
.result-card {
    border-left: 4px solid var(--primary) !important;
    background: rgba(255, 255, 255, 0.02) !important;
    padding: 1.5rem !important;
    border-radius: 0 16px 16px 0 !important;
}
"""

with gr.Blocks(css=CSS) as demo:
    with gr.Column(elem_classes=["container"]):
        # Header Section
        with gr.Column():
            gr.Markdown("# 🛡️ Discri-Net", elem_classes=["title-text"])
            gr.Markdown("ADVANCED MULTIMODAL HATE SPEECH GUARD WITH GEMINI RAG REASONING", elem_classes=["subtitle-text"])

        with gr.Row(equal_height=True):
            # Left Column: Inputs
            with gr.Column(scale=11):
                with gr.Column(elem_classes=["glass-panel"]):
                    gr.Markdown("### 📥 Input Content")
                    input_img = gr.Image(
                        type="pil", 
                        label="Meme / Image", 
                        interactive=True,
                        elem_classes=["input-box"]
                    )
                    input_text = gr.Textbox(
                        label="⌨️ Caption (Meme Text)", 
                        placeholder="Provide text for analysis or leave empty for OCR...",
                        lines=3,
                        elem_classes=["input-box"]
                    )
                    btn = gr.Button("🚀 Start Deep Analysis", variant="primary", elem_classes=["analyze-btn"])
                    
                    with gr.Column(elem_classes=["pro-tip-card"]):
                        gr.Markdown("#### 💡 Intelligence Overview", elem_classes=["pro-tip-title"])
                        gr.Markdown("""
                        - **Multimodal Engine**: Fuses CLIP vision features with MMBT text encoders.
                        - **Dynamic RAG**: Consults 50+ localized safety policies via FAISS vector search.
                        - **Political Guard**: Enhanced logic for identifying political personalities and neutral discourse.
                        - **OCR Pipeline**: Automated text extraction via Gemini Pro Vision.
                        """)

            # Right Column: Outputs
            with gr.Column(scale=13):
                with gr.Column(elem_classes=["glass-panel"]):
                    gr.Markdown("### 📊 AI Analysis Results")
                    output_md = gr.Markdown(
                        "Ready for analysis. Please upload a meme or provide text to begin.",
                        elem_id="results-display"
                    )
                    
        # Footer
        gr.Markdown(
            "<div style='text-align: center; margin-top: 3rem; color: #64748b; font-size: 0.9rem;'>"
            "Powered by Gemini 2.0 & CLIP ViT-Large • Research Edition v2.1"
            "</div>"
        )

        btn.click(
            fn=predict, 
            inputs=[input_img, input_text], 
            outputs=output_md,
            api_name="analyze"
        )

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate"),
        css=CSS
    )
