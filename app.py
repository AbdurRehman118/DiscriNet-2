import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from model import MMBTCLIP
import os
import joblib
import numpy as np

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

# 4. OCR Model (Gemini API)
import google.generativeai as genai

# Hardcoded API Key
GEMINI_KEY = "REDACTED"
genai.configure(api_key=GEMINI_KEY)
print("Initialized Gemini OCR Integration.")

def predict(image, text):
    if image is None: return "Please upload an image."
    
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
        "gemini-2.0-flash-lite-preview-02-05", # Often free/unlimited in preview
        "gemini-2.0-flash-exp",           # Experimental often has separate quota
        "gemini-flash-latest",            # Alias to stable flash
        "gemini-1.5-flash-latest"         # Fallback alias
    ]

    if text.strip() == "":
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

        result_str = (
            f"**Prediction**: {'HATEFUL' if prob_final > 0.5 else 'NON-HATEFUL'}\n"
            f"**Confidence**: {prob_final:.4f} ({source})\n\n"
            f"**Input Text**: {text} {ocr_source}\n\n"
            f"**Breakdown**:\n"
            f"- Base Model Prob: {prob_base:.4f}\n"
            f"- Policy Score (Max Sim): {prob_policy_norm:.4f} (Raw: {prob_policy:.4f})\n\n"
            f"**Matched Policy ({match_source})**:\n> {top_policy}"
        )
        return result_str
    except Exception as e:
        return f"Error during inference: {e}"

# UI Layout
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Meme Image"),
        gr.Textbox(label="Meme Text (Caption, leave empty for OCR)")
    ],
    outputs=gr.Markdown(label="Analysis Result"),
    title="Hateful Memes Detector (MMBT + Policy Stacking)",
    description="Upload a meme to detect hate speech. Uses Gemini 1.5 Flash for OCR if no caption is provided.",
    theme="default"
)

if __name__ == "__main__":
    iface.launch(share=True)
