
import os
from common import constants
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

DATA_PATH = './data/qa_data.txt'
VECTOR_STORE_PATH = './vector_storage/local_vector'

# 创建 Data Loader
loader = TextLoader(DATA_PATH, encoding='UTF-8')

# 创建 Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=15, chunk_overlap=0, separator='\n')

# 创建 Embedding Model
embedding_model = OpenAIEmbeddings(openai_api_key=constants.API_KEY)

# 加载数据
loader_documents = loader.load()

# 切分 Document
documents = text_splitter.split_documents(loader_documents)
print(documents)

# 判断是否已经有本地生成的 Embedding Vector
if os.path.exists(VECTOR_STORE_PATH):
    db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
else:
    # 使用 Embedding Model 生成向量，使用 FAISS 存储向量
    db = FAISS.from_documents(documents, embedding_model)
    # 存储生成向量避免重复生成
    db.save_local(VECTOR_STORE_PATH)

while True:
    query = input("Human: ")
    docs = db.similarity_search(query)
    print(docs)
