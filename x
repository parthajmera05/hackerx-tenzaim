import os
import tempfile
import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_groq import ChatGroq

import uvicorn

# === Load .env and configs ===
load_dotenv()
BEARER_TOKEN = os.getenv("HACKRX_API_KEY", "test123")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Init FastAPI app ===
app = FastAPI()

@app.get("/")
def root():
    return {"message": " HackRx backend is working!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# === Model ===
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# === Request Schema ===
class InputData(BaseModel):
    documents: list[str]  # list of local file paths or URLs
    questions: list[str]

# === PDF Loader ===
def load_pdf(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            path_or_url = tmp.name
    doc = fitz.open(path_or_url)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# === Text Splitter ===
def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200
    )
    return splitter.create_documents([text])

# === Vector Store ===
def embed_documents(docs):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    return vectordb

# === Prompt Template ===
rag_prompt = PromptTemplate.from_template("""
You are an expert insurance analyst. Based only on the context below, answer the user's question.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question:
{question}
""")

# === Build RAG Chain ===
def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    chain = (
        RunnableMap({
            "context": lambda q: retriever.invoke(q),
            "question": RunnablePassthrough()
        })
        | RunnableLambda(lambda d: {
            "context": "\n\n".join(doc.page_content for doc in d["context"]),
            "question": d["question"]
        })
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain

# === Main Endpoint ===
@app.post("/hackrx/run")
async def run_rag(
    payload: InputData,
    authorization: str = Header(...)
):
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        combined_text = ""
        for path in payload.documents:
            combined_text += load_pdf(path) + "\n"

        chunks = split_text(combined_text)
        vectordb = embed_documents(chunks)
        rag_chain = build_rag_chain(vectordb)

        answers = [rag_chain.invoke(q).strip() for q in payload.questions]
        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Run Dev Server ===
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
