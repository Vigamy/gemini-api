from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

llm = genai.GenerativeModel(
    model_name = "gemini-2.5-flash",
    generation_config = genai.types.GenerationConfig(
        temperature = 0.7,
        top_p = 0.95
        # max_output_tokens = 1024
        # stop_sequences = ["\n\n"]
    )
)
try :
    response = llm.generate_content(input("Digite sua pergunta: "))
    
    print(response.text)
except Exception as e:
    print("Ocorreu um erro ao gerar o conteúdo:", str(e))