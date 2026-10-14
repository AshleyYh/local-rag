
def build_context(
    results: list[dict],
    max_chars: int = 8000,
) -> str:
    """
    将检索结果整理为带来源的上下文，并跳过完全重复的文本。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    if not results:
        return ""

    sections = []
    seen_contents = set()
    total_chars = 0

    for index, item in enumerate(results, start=1):
        content = (item.get("content") or "").strip()

        if not content:
            continue

        # 先去除完全重复的文本块
        normalized = " ".join(content.split())
        if normalized in seen_contents:
            continue
        seen_contents.add(normalized)

        source = item.get("source") or "未知文件"
        page = item.get("page")
        page_label = f"第 {page} 页" if page is not None else "页码未知"

        section = (
            f"[资料 {index}]\n"
            f"文件：{source}\n"
            f"位置：{page_label}\n"
            f"内容：\n{content}"
        )

        remaining = max_chars - total_chars
        if remaining <= 0:
            break

        # 避免上下文超过字符预算
        if len(section) > remaining:
            if remaining < 100:
                break
            section = section[:remaining]

        sections.append(section)
        total_chars += len(section)

    return "\n\n".join(sections)
