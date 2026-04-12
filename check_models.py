import google.generativeai as genai
import os

GEMINI_KEY = "AIzaSyBuR64WJMYFMaC2khMxsE5X_iN8VAq9-78"
genai.configure(api_key=GEMINI_KEY)

print("Listing supported models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
