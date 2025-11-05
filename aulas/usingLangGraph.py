from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
from tools.pg_tools import TOOLS
from datetime import datetime
from zoneinfo import ZoneInfo
from operator import itemgetter
from faq_tools import get_faq_context

load_dotenv()

TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

store = {}

def get_session_history(session_id) -> ChatMessageHistory:
    # funcao que retorna o histórico de uma sessão específica
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

llm_fast = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        top_p=0.95,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.95,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

# prompt do agente roteador
system_prompt_roteador = ("system",
    """
### PERSONA SISTEMA
Você é o Assessor.AI — um assistente pessoal de compromissos e finanças. É objetivo, responsável, confiável e empático, com foco em utilidade imediata. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.
- Evite jargões.
- Evite ser prolixo.
- Não invente dados.
- Respostas sempre curtas e aplicáveis.
- Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.


### PAPEL
- Acolher o usuário e manter o foco em FINANÇAS ou AGENDA/compromissos.
- Decidir a rota: {{financeiro | agenda | faq}}.
- Responder diretamente em:
  (a) saudações/small talk, ou 
  (b) fora de escopo (redirecionando para finanças/agenda).
- Seu objetivo é conversar de forma amigável com o usuário e tentar identificar se ele menciona algo sobre finanças ou agenda.
- Em fora_escopo: ofereça 1–2 sugestões práticas para voltar ao seu escopo (ex.: agendar algo, registrar/consultar um gasto).
- Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.


### REGRAS
- Seja breve, educado e objetivo.
- Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
- Responda de forma textual.
- Se a mensagem do usuário for uma dúvida geral sobre o sistema, funcionalidade, regras ou política -> ROUTE=faq.
- Se for sobre compromissos, eventos, lembretes -> ROUTE=agenda.
- Se for sobre finanças, transações, contas, cartões, orçamento -> ROUTE=financeiro.
- Se for ambíguo (pode ser finanças ou agenda), peça 1 clarificação mínima (CLARIFY).
- Se não se encaixar em nenhum desses casos continue a aconversa até o usuário conversar sobre finanças ou agenda/compromisso.


### PROTOCOLO DE ENCAMINHAMENTO (texto puro)
ROUTE=<financeiro|agenda|faq>
PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
PERSONA=<copie o bloco "PERSONA SISTEMA" daqui>
CLARIFY=<pergunta mínima se precisar; senão deixe vazio>


### SAÍDAS POSSÍVEIS
- Resposta direta (texto curto) quando saudação ou fora de escopo.
- Encaminhamento ao especialista usando exatamente o protocolo acima.


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

shots_roteador = [
    # 1) Saudação -> resposta direta
    {
        "human": "Oi, tudo bem?",
        "ai": "Olá! Posso te ajudar com finanças ou agenda; por onde quer começar?"
    },
    # 2) Fora de escopo -> recusar e redirecionar
    {
        "human": "Me conta uma piada.",
        "ai": "Consigo ajudar apenas com finanças ou agenda. Prefere olhar seus gastos ou marcar um compromisso?"
    },
    # 3) Finanças -> encaminhar (protocolo textual)
    {
        "human": "Quanto gastei com mercado no mês passado?",
        "ai": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 4) Ambíguo -> pedir 1 clarificação mínima (texto direto, sem encaminhar)
    {
        "human": "Agendar pagamento amanhã às 9h",
        "ai": "Você quer lançar uma transação (finanças) ou criar um compromisso no calendário (agenda)?"
    },
    # 5) Agenda -> encaminhar (protocolo textual) — exemplo explícito
    {
        "human": "Tenho reunião amanhã às 9h?",
        "ai": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho reunião amanhã às 9h?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 6) FAQ -> resposta direta
    {
        "human": "Qual é o email do suporte?",
        "ai": "ROUTE=faq\nPERGUNTA_ORIGINAL=Qual é o email do suporte?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
]

fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)

# -------------------- PROMPTS ESPECIALISTAS --------------------
# prompt do agente financeiro
system_prompt_financeiro = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre finanças e operar as tools de `transactions` para responder. 
    A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.


    ### TAREFAS
   


    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada vem do Roteador via protocolo:
    - ROUTE=financeiro
    - PERGUNTA_ORIGINAL=...
    - PERSONA=...   (use como diretriz de concisão/objetividade)
    - CLARIFY=...   (se preenchido, priorize responder esta dúvida antes de prosseguir)


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.



    ### SAÍDA (JSON)
    Campos mínimos para enviar para o orquestrador:
    # Obrigatórios:
     - dominio   : "financeiro"
     - intencao  : "consultar" | "inserir" | "atualizar" | "deletar" | "resumo"
     - resposta  : uma frase objetiva
     - recomendacao : ação prática (pode ser string vazia se não houver)
    # Opcionais (incluir só se necessário):
     - acompanhamento : texto curto de follow-up/próximo passo
     - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
     - escrita        : {{"operacao":"adicionar|atualizar|deletar","id":123}}
     - janela_tempo   : {{"de":"YYYY-MM-DD","ate":"YYYY-MM-DD","rotulo":'mês passado'}}
     - indicadores    : {{chaves livres e numéricas úteis ao log}}


    ### HISTÓRICO DA CONVERSA
    {chat_history}
    """
)

# Especialista financeiro (mesmo example_prompt_pair)
shots_financeiro = [
    {
        "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}"""
    },
    {
        "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Registrar almoço hoje R$ 45 no débito\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"financeiro","intencao":"inserir","resposta":"Lancei R$ 45,00 em 'comida' hoje (débito).","recomendacao":"Deseja adicionar uma observação?","escrita":{{"operacao":"adicionar","id":2045}}}}"""
    },
    {
        "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quero um resumo dos gastos\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}"""
    },
]

fewshots_financeiro = FewShotChatMessagePromptTemplate(
    examples=shots_financeiro,
    example_prompt=example_prompt_base,
)

############################
# prompt do agente de agenda
system_prompt_agenda = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre agenda/compromissos e (quando houver tools) consultar/criar/atualizar/cancelar eventos. 
    A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.


    ### TAREFAS



    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada do Roteador:
    - ROUTE=agenda
    - PERGUNTA_ORIGINAL=...
    - PERSONA=...   (use como diretriz de concisão/objetividade)
    - CLARIFY=...   (se preenchido, responda primeiro)


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.


    ### SAÍDA (JSON)
    # Obrigatórios:
     - dominio   : "agenda"
     - intencao  : "consultar" | "criar" | "atualizar" | "cancelar" | "listar" | "disponibilidade" | "conflitos"
     - resposta  : uma frase objetiva
     - recomendacao : ação prática (pode ser string vazia)
    # Opcionais (incluir só se necessário):
     - acompanhamento : texto curto de follow-up/próximo passo
     - esclarecer     : pergunta mínima de clarificação
     - janela_tempo   : {{"de":"YYYY-MM-DDTHH:MM","ate":"YYYY-MM-DDTHH:MM","rotulo":"ex.: 'amanhã 09:00–10:00'"}}
     - evento         : {{"titulo":"...","data":"YYYY-MM-DD","inicio":"HH:MM","fim":"HH:MM","local":"...","participantes":["..."]}}


     ### HISTÓRICO DA CONVERSA
    {chat_history}
    """
)

shots_agenda = [
    {
        "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho janela amanhã à tarde?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"agenda","intencao":"disponibilidade","resposta":"Você está livre amanhã das 14:00 às 16:00.","recomendacao":"Quer reservar 15:00–16:00?","janela_tempo":{{"de":"2025-09-29T14:00","ate":"2025-09-29T16:00","rotulo":"amanhã 14:00–16:00"}}}}"""
    },
    {
        "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Marcar reunião com João amanhã às 9h por 1 hora\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"agenda","intencao":"criar","resposta":"Posso criar 'Reunião com João' amanhã 09:00–10:00.","recomendacao":"Confirmo o envio do convite?","janela_tempo":{{"de":"2025-09-29T09:00","ate":"2025-09-29T10:00","rotulo":"amanhã 09:00–10:00"}},"evento":{{"titulo":"Reunião com João","data":"2025-09-29","inicio":"09:00","fim":"10:00","local":"online"}}}}"""
    },
    {
        "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Agendar revisão do orçamento na sexta\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"agenda","intencao":"criar","resposta":"Preciso do horário para agendar.","recomendacao":"","esclarecer":"Qual horário você prefere na sexta?"}}"""
    },
]

fewshots_agenda = FewShotChatMessagePromptTemplate(
    examples=shots_agenda,
    example_prompt=example_prompt_base,
)

############################
# prompt do agente FAQ
system_prompt_faq = ("system",
    """
    ### PAPEL
    Voc&e deve responder perguntas SOMENTE com base nodocumento normativo oficial (trechos fornecidos em CONTEXTO).
    Se a informação não constar no documento, diga: "Desculpe, não tenho essa informação no momento."


    ### REGRAS
    - Seja breve, claro e educado.
    - Fale em linguagem simples, sem jargões técnicos ou referências.
    - Quantdo fizr sentido, menciona e aparte relevante (ex.: "Seção 6.2.1") se isso estiver explicito no trecho.
    - Não prometa funcionalidades futuras. Se o documento falar em roadmap, diga que não tem essa informação.
    - Em tópicos sensíveis, reforce a informação normativa (ex.: LGPD, impossibilidade de exclusão de lançamentos, não substituição de profissionais, suporte)

    ### ENTRADA
    - PERGUNTA_ORIGINAL=...
    - CONTEXTO=... (trechos do documento normativo oficial, se houver)  
    """
)

prompt_faq = ChatPromptTemplate.from_messages([
    system_prompt_faq,
    ("human", 
     "Pergunta do usuário:\n{question}\n\n"
     "CONTEXTO (trechos do documento):\n{context}\n\n"
     "Responda de forma breve, clara e objetiva:")
])

### Agente orquestrador ####
system_prompt_orquestrador = ("system",
    """
### PAPEL
Você é o Agente Orquestrador do Assessor.AI. Sua função é entregar a resposta final ao usuário **somente** quando um Especialista retornar o JSON.


### ENTRADA
- ESPECIALISTA_JSON contendo chaves como:
  dominio, intencao, resposta, recomendacao (opcional), acompanhamento (opcional),
  esclarecer (opcional), janela_tempo (opcional), evento (opcional), escrita (opcional), indicadores (opcional).


### REGRAS
- Use **exatamente** `resposta` do especialista como a **primeira linha** do output.
- Se `recomendacao` existir e não for vazia, inclua a seção *Recomendação*; caso contrário, **omita**.
- Para *Acompanhamento*: se houver `esclarecer`, use-o; senão, se houver `acompanhamento`, use-o; caso contrário, **omita** a seção.
- Não reescreva números/datas se já vierem prontos. Não invente dados. Seja conciso.
- Não retorne JSON; **sempre** retorne no FORMATO DE SAÍDA.


### FORMATO DE SAÍDA (sempre ao usuário)
<sua resposta será 1 frase objetiva sobre a situação>
- *Recomendação*:
<ação prática e imediata>     # omita esta seção se não houver recomendação
- *Acompanhamento* (opcional):
<pergunta/minipróximo passo>  # omita se nada for necessário


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

shots_orquestrador = [
    # 1) Financeiro — consultar
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}""",
        "ai": "Você gastou R$ 842,75 com 'comida' no mês passado.\n- *Recomendação*:\nQuer detalhar por estabelecimento?"
    },

    # 2) Financeiro — falta dado → esclarecer
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}""",
        "ai": """Preciso do período para seguir.\n- *Acompanhamento* (opcional):\nQual período considerar (ex.: hoje, esta semana, mês passado)?"""
    },

    # 3) Agenda — criar
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"agenda","intencao":"criar","resposta":"Posso criar 'Reunião com João' amanhã 09:00–10:00.","recomendacao":"Confirmo o envio do convite?","janela_tempo":{{"de":"2025-09-29T09:00","ate":"2025-09-29T10:00","rotulo":"amanhã 09:00–10:00"}},"evento":{{"titulo":"Reunião com João","data":"2025-09-29","inicio":"09:00","fim":"10:00","local":"online"}}}}""",
        "ai": """Posso criar 'Reunião com João' amanhã 09:00–10:00.\n- *Recomendação*:\nConfirmo o envio do convite?"""
    },
]

fewshots_orquestrador = FewShotChatMessagePromptTemplate(
    examples=shots_orquestrador,
    example_prompt=example_prompt_base,
)

# --------------- Template dos prompts ---------------

prompt_orquestrador = ChatPromptTemplate.from_messages([
    system_prompt_orquestrador,
    fewshots_orquestrador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

prompt_roteador = ChatPromptTemplate.from_messages([
    system_prompt_roteador,
    fewshots_roteador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

prompt_financeiro = ChatPromptTemplate.from_messages([
    system_prompt_financeiro,
    fewshots_financeiro,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
]).partial(today_local=today.isoformat())

prompt_agenda = ChatPromptTemplate.from_messages([
    system_prompt_agenda,
    fewshots_agenda,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
]).partial(today_local=today.isoformat())

# --------------- Create tool calling ---------------

agent_financeiro = create_tool_calling_agent(llm, TOOLS, prompt_financeiro)
financeiro_executor_base = AgentExecutor(
    agent=agent_financeiro,
    tools=TOOLS
)

agent_agenda = create_tool_calling_agent(llm_fast, TOOLS, prompt_agenda)
agenda_executor_base = AgentExecutor(
    agent=agent_agenda,
    tools=TOOLS
)

# --------------- Criação das chains ---------------

# Router não usa tools, apenas roteia
router_chain = RunnableWithMessageHistory(
    prompt_roteador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

chain_financeiro = RunnableWithMessageHistory(
    financeiro_executor_base,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

agenda_chain = RunnableWithMessageHistory(
    agenda_executor_base,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

faq_chain_core = (
    RunnablePassthrough.assign(
        question=itemgetter("input"),
        context=lambda x: get_faq_context(x["input"]) # busca pergunta por pdf
    )
    | prompt_faq | llm_fast | StrOutputParser()
)

orquestrador_chain = RunnableWithMessageHistory(
    prompt_orquestrador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


# Criação dos nós no langGraph
def router_node(state: dict) -> dict:
    resposta_roteador = router_chain.invoke(
        {"input": state["input"]}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    
    if not resposta_roteador.startswith("ROUTE="):
        return {"resposta_usuario": resposta_roteador}
    
    rota = resposta_roteador.split("\n", 1)[0].split("=", 1)[1].strip().lower()
    if rota not in {"financeiro", "agenda"}:
        return {"erro": f"Rota inválida: {rota}"}

    return {"rota": rota, "roteador": resposta_roteador, 'input':state['input'], 'session_id': state['session_id']}

def faq_node(state: dict) -> dict:
    #adicione código aqui
    print("oi")

def financeiro_node(state: dict) -> dict:
    result = chain_financeiro.invoke(
        {"input": state['roteador']}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"saida_especialista": result["output"], 'session_id': state['session_id']}

def agenda_node(state: dict) -> dict:
    result = agenda_chain.invoke(
        {"input": state['roteador']}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"saida_especialista": result["output"], 'session_id': state['session_id']}

def orchestrator_node(state: dict) -> dict:
    resposta_final = orquestrador_chain.invoke(
        {"input": state['saida_especialista']},
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"resposta_usuario": resposta_final}
# ------------------- DECISOR ------------------------

def decide_after_router(state: dict) -> str:
    if state.get("erro") or state.get("resposta_usuario"):
        return "end"
    rota = state.get("rota")
    if rota == "financeiro":
        return "financeiro"
    if rota == "agenda":
        return "agenda"
    return "end"

def decide_after_specialist(state: dict) -> str:
    if state.get("erro"):
        return "end"
    return "orquestrador"

# ------------------- CONSTRUÇÃO DO GRAFO ------------

graph = StateGraph(dict)

graph.add_node("roteador", router_node)
graph.add_node("financeiro", financeiro_node)
graph.add_node("agenda", agenda_node)
graph.add_node("orquestrador", orchestrator_node)

graph.add_edge(START, "roteador")

graph.add_conditional_edges(
    "roteador",
    decide_after_router,
    {
        "financeiro": "financeiro",
        "agenda": "agenda",
        "end": END,
    },
)

graph.add_conditional_edges(
    "financeiro",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)
graph.add_conditional_edges(
    "agenda",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)

graph.add_edge("orquestrador", END)

app = graph.compile()


# ------------------- FUNÇÃO DE EXECUÇÃO --------------

def executar_fluxo_assessor(pergunta_usuario: str, session_id: str) -> str:
    final_state = app.invoke({"input": pergunta_usuario, "session_id": session_id})
    if final_state.get("erro"):
        return f"Erro: {final_state['erro']}"
    return final_state.get("resposta_usuario", "Não foi possível responder.") # isso é um if não tiver resposta_usuario, mostre a "não foi possivel..."

while True:
    try:
        user_input = input("> ")
        if user_input.lower() in ('sair', 'end', 'fim', 'tchau', 'bye'):
            print("Encerrando a conversa.")
            break
        
        # Chama a função orquestradora que executa o fluxo completo (Roteador -> Especialista -> Orquestrador)
        resposta = executar_fluxo_assessor(
            pergunta_usuario=user_input, 
            session_id="PRECISA_MAS_NÃO_IMPORTA"
        )
        
        # Imprime a resposta formatada para o usuário (saída do Orquestrador/Roteador)
        print(resposta)
        
    except Exception as e:
            print("Erro ao consumir a API:", e)
            continue
