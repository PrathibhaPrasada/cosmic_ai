import os
from dotenv import load_dotenv
import google.generativeai as genai

#Load environment variables from .env
load_dotenv()

#Configure Gemini API with your key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#Use lightweight model for free-tier usage
model = genai.GenerativeModel("gemini-1.5-flash")

#Get user input from terminal
user_input = input("🌌 Ask Cosmic_ai something about space: ")

#Generate content
response = model.generate_content(user_input)

#Print the response
print("\n🌌 Cosmic_ai says:\n")
print(response.text)
