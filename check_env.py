import sys
import os

print(f"--- Environment Status ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"--------------------------")

modules = [
    "gradio", "torch", "transformers", "PIL", "joblib",
    "numpy", "langchain_google_genai", "google.generativeai",
    "dotenv", "peft", "unsloth_zoo"
]

for m in modules:
    try:
        if m == "PIL":
            import PIL.Image
            ver = getattr(PIL, "__version__", "unknown")
        else:
            mod = __import__(m)
            ver = getattr(mod, "__version__", "unknown")
            if m == "google.generativeai":
                import google.generativeai as genai
                ver = genai.__version__
        
        print(f"[OK] {m:20} | Version: {ver}")
    except ImportError:
        print(f"[MISSING] {m:20}")
    except Exception as e:
        print(f"[ERROR] {m:20} | {e}")
