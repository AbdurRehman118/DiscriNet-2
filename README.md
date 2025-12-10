# Hateful Memes Detector (MMBT-CLIP + Dynamic Policy Gating)

A robust, multimodal hate speech detection system designed for the Facebook Hateful Memes dataset. It combines a state-of-the-art **MMBT-CLIP** architecture with a novel **Dynamic Policy Gating** mechanism and high-accuracy **Gemini OCR**.

---

## 🚀 Key Features

### 1. **MMBT-CLIP Architecture** (ViT-Large)
- **Visual Encoder**: `openai/clip-vit-large-patch14`
- **Fusion**: Multimodal Bi-Transformer (MMBT) effectively combines image and text embeddings.
- **Optimization**: Uses Focal Loss for hard example mining and LoRA adapters for efficient fine-tuning on consumer GPUs (RTX 3060 supported).

### 2. **Dynamic Policy Gating** ("Knowledge Ensembling")
Instead of a simple black-box classifier, the system references a database of known hateful policies/narratives.
- **Retrieval**: Finds the most similar policy using CLIP embedding similarity.
- **Gating**: If the similarity exceeds a learned threshold (`0.28`), the policy score is dynamically boosted in the final prediction.
- **Benefit**: Increases sensitivity to dog whistles and coded language.

### 3. **Smart OCR (Gemini API)**
- **Engine**: Uses **Google Gemini 2.0 Flash / 1.5 Flash** for superior text extraction compared to standard OCR tools.
- **Robustness**: Implements a smart fallback loop. If the primary model hits a Free Tier quota limit (`429`), it automatically switches to alternative models (`Lite`, `Experimental`) to ensure continuous operation.
- **Transparency**: Hardcoded API configuration for seamless "plug-and-play" testing.

---

## 🛠️ Installation

1. **Environment Setup**:
   ```powershell
   conda create -n hateful_memes python=3.10
   conda activate hateful_memes
   pip install -r env.txt
   ```

2. **Dataset**:
   Place the Facebook Hateful Memes JSONL files in `datasets/Facebook Memes/data/`.

---

## 💻 Usage

### 1. Interactive Web Interface (Inference)
Launch the Gradio UI to test the model with your own memes.
```powershell
python app.py
```
- **OCR**: Leave the "Caption" field empty. The app will use Gemini to extract text automatically.
- **Policy**: The result will show which policy was matched and if it triggered a "Policy Boost".

### 2. Training
To retrain the model on the Facebook Memes dataset:
```powershell
./run_facebook.ps1
```
*(See `run_facebook.ps1` for detailed hyperparameters like batch size, learning rate, and epochs)*

### 3. Analysis
Generate performance metrics and ROC curves:
```powershell
python analyze_results.py
```
Outputs are saved to `results/facebook_analysis_dynamic/`.

---

## 📊 Performance
- **ROC-AUC**: ~0.766 (Dev Set)
- **Accuracy**: ~71.4% (Dynamic Gating)
- **F1 Score**: ~0.730

---

## 📂 Project Structure
- `app.py`: Gradio Web UI with Gemini OCR & Policy logic.
- `model.py`: MMBT-CLIP model definition.
- `train.py`: Main training loop with Focal Loss & Gradient Accumulation.
- `infer_with_policy.py`: Offline inference script.
- `policies/`: Database of policy texts for retrieval.
- `datasets/`: Data storage.

---

**Note**: This project uses the Facebook Hateful Memes dataset. Ensure you comply with its license terms.


