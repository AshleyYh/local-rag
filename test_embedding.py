
from app.services.embedding_service import embed_texts

texts = [
    "医疗机构应当遵守医保相关规定。",
    "定点医疗机构需要按要求报送医疗费用数据。",
    "Python 可以使用列表推导式处理数据。",
]

vectors = embed_texts(texts)

print("输入文本数量：", len(texts))
print("向量数量：", len(vectors))
print("第一个向量的维度：", len(vectors[0]))
print("第一个向量的前 5 个值：", vectors[0][:5])

assert len(vectors) == len(texts)
assert all(len(vector) == 1024 for vector in vectors)

print("Embedding 测试成功！")
