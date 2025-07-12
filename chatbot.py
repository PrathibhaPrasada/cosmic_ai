import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env...
load_dotenv()
api_key = os.getenv("GEMINIAI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Use Gemini Pro (text-only)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

print("Welcome to Cosmic.ai (Gemini version) 🌌")
print("(Type 'exit' to quit)\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ['exit', 'quit']:
        print("Goodbye! 🚀")
        break
    try:
        response = model.generate_content(prompt)
        print("Cosmic.ai:", response.text)
    except Exception as e:
        print("⚠️ Error:", e)
