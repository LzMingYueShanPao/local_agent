import os
from common import constants
from langchain.embeddings import OpenAIEmbeddings

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

# 将文本转换为embedding
embedding_model = OpenAIEmbeddings(openai_api_key=constants.API_KEY)

# 对单个query进行编码
embedded_query = embedding_model.embed_query("销量最高的手机品牌")
# 1536
print(len(embedded_query))
# 对文档进行编码
embeddings = embedding_model.embed_documents(
    [
        "销量最高的手机品牌",
        "哪个牌子的手机卖的最好"
    ]
)
# 2
print(len(embeddings))
# 1536
print(len(embeddings[0]))
# [[-0.02064775016628096, -0.011199799833597309, 0.0009419533118609369, ... ]]
print(embeddings)