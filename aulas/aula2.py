from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

llm = genai.GenerativeModel(
    model_name = "gemini-2.5-flash",
    system_instruction="veja a pergunta e procure a resposta correta. Siga uma lógica meio fora da caixinha",
    generation_config = genai.types.GenerationConfig(
        temperature = 0.7,
        top_p = 0.95
        # max_output_tokens = 1024
        # stop_sequences = ["\n\n"]
    )
)

user_prompt = """
Se ARI é meu pai, e BRUNO é meu primo, então CAROLINA é?:
a)Mãe
b)Prima
c)Tia
d)Sobrinha
e)Irmã
"""

try :
    response = llm.generate_content(user_prompt)
    
    print(response.text)
except Exception as e:
    print("Ocorreu um erro ao gerar o conteúdo:", str(e))