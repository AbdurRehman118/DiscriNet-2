import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY_1")
if not GEMINI_KEY:
    raise ValueError("No GEMINI_API_KEY_1 found in .env")
genai.configure(api_key=GEMINI_KEY)

print("Listing supported models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
