
from app.services.embedding_service import embed_texts
from app.services.vector_store_service import search_similar_chunks


def retrieve_documents(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[dict]:
    """将用户问题转换成向量，检索并整理相关文本块。"""

    query = query.strip()

    if not query:
        raise ValueError("检索问题不能为空")

    if len(query) > 2000:
        raise ValueError("检索问题不能超过 2000 个字符")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k 必须在 1 到 20 之间")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity 必须在 -1 到 1 之间")

    vectors = embed_texts([query])

    if len(vectors) != 1:
        raise RuntimeError("问题向量生成失败")

    chunks = search_similar_chunks(
        query_vector=vectors[0],
        top_k=top_k,
        min_similarity=min_similarity,
    )

    results = []

    for chunk in chunks:
        metadata = chunk["metadata"] or {}

        results.append({
            "chunk_id": chunk["id"],
            "content": chunk["content"],
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": round(chunk["distance"], 6),
            "similarity": round(chunk["similarity"], 6),
        })

    return results
