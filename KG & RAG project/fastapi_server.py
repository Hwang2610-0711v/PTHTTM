"""
REST API để deploy (mục 5.6).
Chạy: uvicorn fastapi_server:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.rag_pipeline import load_vectorstore, rag_query
from config import CHROMA_DIR

app = FastAPI(
    title="Chapter 5 - KG & RAG API",
    description="API cho hệ thống Tiểu đường & Tài chính",
    version="1.0.0",
)


# =========================================================
# SCHEMAS
# =========================================================
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    sources: List[str]


# =========================================================
# LOAD VECTOR STORES
# =========================================================
_vs_cache = {}


def get_vs(name: str):
    if name not in _vs_cache:
        if not (CHROMA_DIR / name).exists():
            raise HTTPException(400, f"Vector store '{name}' chưa tồn tại. Hãy build trước.")
        _vs_cache[name] = load_vectorstore(name)
    return _vs_cache[name]


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Chapter 5 - KG & RAG API",
        "endpoints": [
            "POST /predict/diabetes",
            "POST /predict/finance",
        ],
    }


@app.post("/predict/diabetes", response_model=QueryResponse)
def predict_diabetes(req: QueryRequest):
    vs = get_vs("diabetes")
    result = rag_query(vs, req.question, k=req.top_k)
    return result


@app.post("/predict/finance", response_model=QueryResponse)
def predict_finance(req: QueryRequest):
    vs = get_vs("finance")
    result = rag_query(vs, req.question, k=req.top_k)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)