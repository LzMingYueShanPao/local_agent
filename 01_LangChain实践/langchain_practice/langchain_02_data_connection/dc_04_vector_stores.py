
import os
from common import constants
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 创建编码模型
embedding_model = OpenAIEmbeddings(openai_api_key=constants.API_KEY)

# 加载数据并进行切分
raw_documents = TextLoader('./data/data_vector_stores.txt', encoding='UTF-8').load()
text_splitter = CharacterTextSplitter(chunk_size=15, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(raw_documents)

print(documents)
# [Document(page_content='中国的首都是北京', metadata={'source': './data/data_vector_stores.txt'}),
# Document(page_content='中国的经济中心是上海', metadata={'source': './data/data_vector_stores.txt'}),
# Document(page_content='河北省的省会是石家庄市', metadata={'source': './data/data_vector_stores.txt'}),
# Document(page_content='世界上最高的山峰是珠穆朗玛峰', metadata={'source': './data/data_vector_stores.txt'})]

# 构建向量数据库
db = FAISS.from_documents(documents, embedding_model)

query = "中国的经济中心是哪里？"
docs = db.similarity_search(query)
print(docs[0].page_content)
# 中国的经济中心是上海
