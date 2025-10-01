from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

store = {}
def get_session_history(session_id) -> ChatMessageHistory:
    # funcao que retorna o histórico de uma sessão específica
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.95,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

system_prompt = system_prompt = ("system",
    """
### PERSONA
Você é o Assessor.AI — um assistente pessoal de compromissos e finanças. Você é especialista em gestão financeira e organização de rotina. Sua principal característica é a objetividade e a confiabilidade. Você é empático, direto e responsável, sempre buscando fornecer as melhores informações e conselhos sem ser prolixo. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.


### TAREFAS
- Processar perguntas do usuário sobre finanças, agenda, tarefas, etc.
- Identificar conflitos de agenda e alertar o usuário sobre eles.
- Analise entradas, gastos, dívidas e compromissos informados pelo usuário.
- Responder a perguntas com base nos dados passados e histórico.
- Oferecer dicas personalizadas de gestão financeira.
- Consultar histórico de decisões/gastos/agenda quando relevante.
- Lembrar pendências/tarefas e propor avisos.


### REGRAS
- Resumir entradas, gastos, dívidas, metas e saúde financeira.
- Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
- Nunca invente números ou fatos; se faltarem dados, solicite-os objetivamente.
- Seja direto, empático e responsável; 
- Evite jargões.
- Mantenha respostas curtas e utilizáveis.
- Não salve as informações de valor dos shots, eles são meramente ilustrativos.


### FORMATO DE RESPOSTA
- <sua resposta será 1 frase objetiva sobre a situação>
- *Recomendação*: 
<ação prática e imediata>
- *Acompanhamento* (opcional): 
<se não tiver informações suficientes para fornecer uma resposta curta, se tiver varias respostas possíveis ou se verificar que o pedido do usuário pode ou precisa ser armazenado seu histórico> 


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

shots = [
    # ================ FEW-SHOTS ================
    # 1) Decisão de compra (finanças)
    {"human": 
    "Recebi R$3.000 este mês, já gastei R$2.400 e tenho R$200 de dívida no cartão. Posso comprar um celular de R$1.200?",
    "ai":
     "- Sua sobra atual é ~R$600 e há dívida ativa de R$200.\n"
     "- *Recomendação*: \nNão recomendo a compra agora; priorize quitar a dívida e manter reserva mínima para o mês.\n"
    },

    # 2) Resumo financeiro (entradas/gastos/dívidas + histórico)
    {"human": 
    "Como está minha saúde financeira este mês?",
    "ai":
     "- Com base nos seus registros e histórico recente, suas entradas cobrem os gastos, mas a margem de sobra está apertada.\n"
     "- *Recomendação*: \nReduza 10–15% nas despesas variáveis desta semana e reforce a meta de poupança mínima.\n"
    },

    # 3) Agenda e conflitos
    {"human": "Quais compromissos tenho amanhã? Estou livre para marcar uma call às 15h?",
    "ai":
     "- Há duas reuniões agendadas; identifico potencial conflito por proximidade entre 14h e 16h.\n"
     "- *Recomendação*: \nEvite marcar às 15h; melhor janela: 16h30–17h30.\n"
     "- *Acompanhamento*: \nQuer que eu envie convite para 16h30 e ative lembrete por WhatsApp 30 min antes?"
    },

    # 4) Pendências e histórico (tarefas + notificação)
    {"human": "Estou devendo pagar o IPVA e revisar o orçamento; pode me lembrar?",
    "ai":
     "- Encontrei pendências similares no seu histórico deste período.\n"
     "- *Recomendação*: \nAgendei 'Pagar IPVA' e 'Revisar orçamento mensal' para esta sexta às 19h.\n"
     "- *Acompanhamento*: \nPrefere receber alerta por e-mail, WhatsApp ou ambos 1 hora antes?"
    },
]

fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt
)

prompt = ChatPromptTemplate.from_messages([
    system_prompt,                          # system prompt
    fewshots,                               # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{usuario}")                  # user prompt
])

# Cadeia: prompt -> llm -> Parser (LCEL)
base_chain = prompt | llm | StrOutputParser()

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history=get_session_history,
    input_messages_key="usuario",
    history_messages_key="chat_history"
)

while True:
    user_input = input("> ")
    if user_input.lower() in ('sair', 'exit', 'quit', 'tchau'):
        break
    try:
        resposta = chain.invoke(
            {"usuario": user_input},
            config={"configurable": {"session_id": "PRECISA_MAS_NÃO_IMPORTA"}}
        )
        print(resposta)
    except Exception as e:
        print("Erro ao consumir a API: ", e)

try:
    print(chain.invoke({"usuario": input("Digite uma pergunta: ")}))
except Exception as e:
    print("Ocorreu um erro ao gerar o conteúdo:", str(e))
