
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pdf_pages(
    pages: list[dict],
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    将 PDF 的逐页文本转换为 LangChain Document 并切块。

    pages 示例：
    [
        {"page": 1, "text": "第一页内容"},
        {"page": 2, "text": "第二页内容"},
    ]
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    # 1. 每一页先转换为 LangChain Document
    documents = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page["page"],
                },
            )
        )

    # 2. 创建文本切块器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    # 3. 切块并继承原始元数据
    chunks = splitter.split_documents(documents)

    # 4. 为每个文本块添加编号
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks
