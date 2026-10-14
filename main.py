
from fastapi import FastAPI

from app.api.documents import router as documents_router

from app.api.chat import router as chat_router

app = FastAPI(
    title="Local RAG Knowledge Base",
    description="基于 PDF 的本地知识库问答系统",
    version="0.2.0",
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/")
def index():
    return {"message": "Local RAG is running"}


@app.get("/health")
def health():
    return {
        "status": "UP",
        "service": "local-rag",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)