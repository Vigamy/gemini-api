

from google import genai

import os

api_key = "AIzaSyBBYP9pX4193MEZ4wCOQcy939xeJjtr6-g"

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words", api_key=api_key
)

print(response.text)