
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pdf_service import extract_pdf_text
from app.services.chunk_service import split_pdf_pages
from app.services.embedding_service import embed_texts
from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval_service import retrieve_documents

router = APIRouter(prefix="/documents", tags=["文档管理"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. 检查文件名
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="只允许上传 PDF 文件",
        )

    try:
        # 2. 最多读取 20 MB + 1 字节，用于判断是否超限
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="上传文件不能为空",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="文件不能超过 20 MB",
            )

        # 3. 检查 PDF 文件头
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="文件内容不是有效的 PDF 格式",
            )

        # 4. 提取文本
        result = extract_pdf_text(content)

        # 5. 返回解析结果
        return {
            "filename": file.filename,
            "file_size": len(content),
            "page_count": result["page_count"],
            "text_length": len(result["text"]),
            "preview": result["text"][:1000],
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="PDF 解析失败，请检查文件是否损坏或受保护",
        ) from exc

    finally:
        await file.close()


@router.post("/chunk-preview")
async def preview_pdf_chunks(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="只允许上传 PDF 文件",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="上传文件不能为空",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="文件不能超过 20 MB",
            )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="文件内容不是有效的 PDF 格式",
            )

        # 第一步：解析 PDF
        result = extract_pdf_text(content)

        # 第二步：切块
        chunks = split_pdf_pages(
            pages=result["pages"],
            filename=file.filename,
            chunk_size=500,
            chunk_overlap=100,
        )

        # 第三步：返回切块结果，便于调试
        return {
            "filename": file.filename,
            "page_count": result["page_count"],
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_index": chunk.metadata["chunk_index"],
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "length": len(chunk.page_content),
                    "content": chunk.page_content,
                }
                for chunk in chunks[:20]
            ],
            "truncated": len(chunks) > 20,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="PDF 解析或切块失败",
        ) from exc

    finally:
        await file.close()


@router.post("/embedding-preview")
async def preview_pdf_embeddings(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只允许上传 PDF 文件")

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(status_code=400, detail="上传文件不能为空")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="文件不能超过 20 MB")

        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF 格式")

        # 1. 解析 PDF
        result = extract_pdf_text(content)

        # 2. 切块
        chunks = split_pdf_pages(
            pages=result["pages"],
            filename=file.filename,
            chunk_size=500,
            chunk_overlap=100,
        )

        if not chunks:
            raise HTTPException(status_code=422, detail="PDF 没有可向量化的文本")

        # 3. 为了控制本次测试的调用量，只向量化前三个文本块
        sample_chunks = chunks[:3]
        vectors = embed_texts([
            chunk.page_content for chunk in sample_chunks
        ])

        # 4. 返回向量基本信息，不返回完整向量
        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "embedded_chunks": len(vectors),
            "vector_dimension": len(vectors[0]),
            "results": [
                {
                    "chunk_index": chunk.metadata["chunk_index"],
                    "page": chunk.metadata["page"],
                    "text_preview": chunk.page_content[:100],
                    "vector_dimension": len(vector),
                    "vector_preview": vector[:5],
                }
                for chunk, vector in zip(sample_chunks, vectors)
            ],
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="PDF 向量化失败，请检查 Embedding 配置和云端接口",
        ) from exc

    finally:
        await file.close()




class DocumentSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="需要检索的问题",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回的相关文本块数量",
    )
    min_similarity: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="最低余弦相似度阈值",
    )


@router.post("/search", summary="语义检索文档")
def search_documents(request: DocumentSearchRequest):
    """
    根据用户问题，从已入库的文档中检索相关文本。
    """
    try:
        results = retrieve_documents(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
