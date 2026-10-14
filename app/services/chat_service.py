
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from app.services.context_service import build_context
from app.services.retrieval_service import retrieve_documents

# 加载项目根目录中的 .env 配置
load_dotenv()

_client = None


def get_llm_client() -> OpenAI:
    """获取 DeepSeek 的 OpenAI 兼容客户端。"""
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )

    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        max_retries=2,
    )
    return _client


def ask_question(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> dict:
    """检索知识库，并生成带有可映射引用编号的回答。"""
    query = query.strip()

    if not query:
        raise ValueError("问题不能为空")

    results = retrieve_documents(
        query=query,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    if not results:
        return {
            "question": query,
            "answer": "知识库中没有检索到足够相关的资料，暂时无法根据现有文档回答。",
            "sources": [],
            "retrieved_count": 0,
        }

    # 为每个检索分块分配唯一编号
    source_map = {}
    context_sections = []

    for index, item in enumerate(results, start=1):
        source_id = str(index)
        source = item.get("source") or "未知文件"
        page = item.get("page")
        chunk_id = item.get("chunk_id")

        source_map[source_id] = {
            "citation_id": source_id,
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "similarity": item.get("similarity"),
        }

        page_label = f"第 {page} 页" if page is not None else "页码未知"

        context_sections.append(
            f"[{source_id}]\n"
            f"文件：{source}\n"
            f"位置：{page_label}\n"
            f"分块 ID：{chunk_id}\n"
            f"内容：\n{item.get('content', '')}"
        )

    context = "\n\n".join(context_sections)

    system_prompt = """
你是一个严谨的知识库问答助手。

规则：
1. 只能依据提供的知识库资料回答，不得编造事实。
2. 如果资料不足，明确说明无法从现有资料确认。
3. 对关键事实、数字和结论，在句末标注引用编号，例如 [1] 或 [2]。
4. 引用编号必须来自知识库资料开头的编号。
5. 不得自行编造编号，不要直接编造文件名或页码作为引用。
6. 如果多个资料支持同一结论，可以同时引用，例如 [1][2]。
7. 回答应清晰、直接；资料中的限制和争议应如实说明。
"""

    user_prompt = f"""
请回答以下问题。

【用户问题】
{query}

【知识库资料】
{context}

请在每项重要结论后标注对应的资料编号。
"""

    model = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    client = get_llm_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    answer = (response.choices[0].message.content or "").strip()

    # 提取模型实际输出的引用编号
    cited_ids = set(re.findall(r"\[(\d+)\]", answer))

    # 模型引用了哪些编号
    valid_cited_ids = cited_ids & source_map.keys()

    # 模型引用了哪些不存在的编号
    invalid_citation_ids = sorted(
        cited_ids - source_map.keys(),
        key=int,
    )

    citation_valid = len(invalid_citation_ids) == 0

    # 只返回能够映射到真实检索分块的引用
    cited_sources = [
        source_map[source_id]
        for source_id in sorted(cited_ids, key=int)
        if source_id in source_map
    ]

    # 如果模型没有生成有效引用，保留检索来源供排查，
    # 但不能把它们误认为模型已经引用的来源。
    return {
        "question": query,
        "answer": answer,
        "sources": cited_sources,
        "retrieved_count": len(results),
        "citation_validation": {
            "valid": citation_valid,
            "invalid_ids": invalid_citation_ids,
        },
    }


