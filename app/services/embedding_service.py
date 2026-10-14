
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL_NAME = os.getenv(
    "DASHSCOPE_EMBEDDING_MODEL",
    "text-embedding-v3",
)

if not API_KEY:
    raise RuntimeError(
        "请在项目根目录的 .env 文件中配置 DASHSCOPE_API_KEY"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60.0,
    max_retries=2,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    将多个文本转换为向量。

    每个输入文本对应一个浮点数向量。
    """
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("待向量化的文本不能为空")

    vectors: list[list[float]] = []

    # text-embedding-v3 的批量输入最多处理 10 条，
    # 因此这里按每批 10 条请求。
    for start in range(0, len(texts), 10):
        batch = texts[start:start + 10]

        response = client.embeddings.create(
            model=MODEL_NAME,
            input=batch,
            dimensions=1024,
            encoding_format="float",
        )

        # 根据返回的 index 排序，确保向量与输入文本对应
        results = sorted(response.data, key=lambda item: item.index)

        if len(results) != len(batch):
            raise RuntimeError("Embedding 返回数量与输入文本数量不一致")

        vectors.extend(item.embedding for item in results)

    return vectors
