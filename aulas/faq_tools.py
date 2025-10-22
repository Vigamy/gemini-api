import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Caminho do PDF
PDF_PATH = r"/Users/willamy/Documents/IA-Grilo/aulas/FAQ_assessor_v1.pdf"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    transport="rest"
)


def get_faq_context(question: str):
    """
    Busca os trechos mais relevantes do documento FAQ com base na pergunta.
    Retorna o texto concatenado dos trechos mais similares.
    """
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(question, k=6)

    context_text = "\n\n".join([r.page_content for r in results])

    return context_text