
import json
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


# ==================== 数据库连接配置 ====================

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
    "port": int(os.getenv("POSTGRES_PORT", "5433")),
    "dbname": os.getenv("POSTGRES_DB", "local_rag"),
    "user": os.getenv("POSTGRES_USER", "raguser"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}


def get_connection():
    """获取 PostgreSQL 数据库连接，并注册 pgvector 类型适配。"""
    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


# ==================== 数据表初始化 ====================

def init_vector_table():
    """创建 pgvector 扩展和文档向量表。"""
    with get_connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                embedding VECTOR(1024) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

    return "向量表初始化成功"


# ==================== 文档向量入库 ====================

def save_document_chunks(chunks, vectors):
    """
    批量保存文本块及其 Embedding 向量。

    Args:
        chunks: LangChain Document 列表。
        vectors: 与文本块一一对应的 1024 维向量列表。

    Returns:
        实际写入的文本块数量。
    """
    if len(chunks) != len(vectors):
        raise ValueError("文本块数量与向量数量不一致")

    if not chunks:
        return 0

    rows = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        if len(vector) != 1024:
            raise ValueError(
                f"第 {index} 个向量维度错误："
                f"期望 1024，实际 {len(vector)}"
            )

        rows.append((
            chunk.page_content,
            json.dumps(chunk.metadata, ensure_ascii=False),
            vector,
        ))

    sql = """
        INSERT INTO document_chunks (content, metadata, embedding)
        VALUES (%s, %s::jsonb, %s)
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql, rows)

    return len(rows)



def search_similar_chunks(
    query_vector: list[float],
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[dict]:
    """按余弦相似度检索文本块，并过滤低相似度结果。"""

    if len(query_vector) != 1024:
        raise ValueError(
            f"问题向量维度必须为 1024，实际为 {len(query_vector)}"
        )

    if not 1 <= top_k <= 20:
        raise ValueError("top_k 必须在 1 到 20 之间")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity 必须在 -1 到 1 之间")

    sql = """
        SELECT
            id,
            content,
            metadata,
            embedding <=> %s::vector AS distance
        FROM document_chunks
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    query_vector,
                    query_vector,
                    min_similarity,
                    query_vector,
                    top_k,
                ),
            )
            rows = cursor.fetchall()

    results = []

    for chunk_id, content, metadata, distance in rows:
        distance = float(distance)

        results.append({
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "distance": distance,
            "similarity": 1.0 - distance,
        })

    return results

