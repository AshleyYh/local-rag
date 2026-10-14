
from app.services.retrieval_service import retrieve_documents
from app.services.context_service import build_context


def main():
    query = "低钠盐替换普通盐，能降低多少中风和死亡风险，成本多少，证据等级是什么"

    results = retrieve_documents(
        query=query,
        top_k=5,
        min_similarity=0.5,
    )

    print(f"检索到 {len(results)} 条结果")

    context = build_context(
        results,
        max_chars=8000,
    )

    if not context:
        print("没有找到足够相关的资料")
        return

    print("\n========== RAG 上下文 ==========\n")
    print(context)


if __name__ == "__main__":
    main()
