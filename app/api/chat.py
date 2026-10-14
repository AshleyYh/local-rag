
from fastapi import APIRouter, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.services.chat_service import ask_question

router = APIRouter(prefix="/chat", tags=["智能问答"])


class ChatAskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户提出的问题",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最多检索多少个文本分块",
    )
    min_similarity: float = Field(
        default=0.5,
        ge=-1.0,
        le=1.0,
        description="最低向量相似度阈值",
    )


@router.post("/ask", summary="基于知识库回答问题")
def chat_ask(request: ChatAskRequest):
    try:
        return ask_question(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except OpenAIError as exc:
        # API 密钥、模型名称、网络或上游接口异常等
        raise HTTPException(
            status_code=502,
            detail="调用大模型失败，请检查 DeepSeek 配置和服务状态。",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
