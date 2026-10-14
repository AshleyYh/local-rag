
from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(content: bytes) -> dict:
    """
    从 PDF 二进制内容中提取文本。

    Args:
        content: PDF 文件的二进制内容

    Returns:
        页数和逐页提取的文本
    """
    reader = PdfReader(BytesIO(content))

    if reader.is_encrypted:
        raise ValueError("暂不支持加密 PDF，请先解除文件密码")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text.strip(),
        })

    full_text = "\n\n".join(
        page["text"] for page in pages if page["text"]
    )

    if not full_text.strip():
        raise ValueError(
            "没有提取到文本，文件可能是扫描版 PDF，需要 OCR 识别"
        )

    return {
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
    }
