
import os
from common import constants
from langchain.embeddings import OpenAIEmbeddings

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

embedding_model = OpenAIEmbeddings(openai_api_key=constants.API_KEY)

# 对单个query进行编码
embedded_query = embedding_model.embed_query("销量最高的手机品牌")
print(len(embedded_query))

# 对文档进行编码
embeddings = embedding_model.embed_documents(
    [
        "销量最高的手机品牌",
        "哪个牌子的手机卖的最好"
    ]
)
print(len(embeddings))
print(len(embeddings[0]))
