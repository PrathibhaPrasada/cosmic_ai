import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()
print("✅ Models available to your key:")
for model in models:
    print(f"- {model.name}")
