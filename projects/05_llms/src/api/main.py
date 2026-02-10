from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from src.rag.pipeline import RAGPipeline

app = FastAPI(
    title="LLM RAG Document Assistant",
    version="1.0.0"
)

DATA_DIR = Path("data/raw")
rag_pipeline = RAGPipeline(data_dir=DATA_DIR)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest_documents():
    rag_pipeline.ingest()
    return {"status": "Documents indexed successfully"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = rag_pipeline.ask(request.question)
    return {"answer": answer}

