
import sys
from pathlib import Path

from pypdf import PdfReader

from app.services.chunk_service import split_pdf_pages
from app.services.embedding_service import embed_texts
from app.services.vector_store_service import save_document_chunks


def ingest_pdf(pdf_path: str):
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("目前只支持 PDF 文件")

    print(f"1. 开始解析 PDF：{path.name}")

    reader = PdfReader(str(path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_num,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "PDF 中没有提取到文本，可能是扫描版 PDF，需要 OCR"
        )

    print(f"   PDF 总页数：{len(reader.pages)}")
    print(f"   有效文本页数：{len(pages)}")

    print("2. 开始切块")

    chunks = split_pdf_pages(
        pages=pages,
        filename=path.name,
        chunk_size=500,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError("切块结果为空")

    print(f"   切块数量：{len(chunks)}")
    print(f"   首块预览：{chunks[0].page_content[:100]}")

    print("3. 开始生成 Embedding")

    texts = [chunk.page_content for chunk in chunks]
    vectors = embed_texts(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding 数量与文本块数量不一致")

    print(f"   向量数量：{len(vectors)}")
    print(f"   向量维度：{len(vectors[0])}")

    print("4. 开始写入 PostgreSQL + pgvector")

    saved_count = save_document_chunks(chunks, vectors)

    print(f"   成功写入：{saved_count} 条")
    print("5. 文档入库完成")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            '用法：python test_ingest.py "C:\\Users\yh200\\Desktop\\HowToLiveBetter(1).pdf"'
        )
        sys.exit(1)

    try:
        ingest_pdf(sys.argv[1])
    except Exception as exc:
        print(f"入库失败：{type(exc).__name__}: {exc}")
        raise
