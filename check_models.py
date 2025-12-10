import google.generativeai as genai
import os

GEMINI_KEY = "REDACTED"
genai.configure(api_key=GEMINI_KEY)

print("Listing supported models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
