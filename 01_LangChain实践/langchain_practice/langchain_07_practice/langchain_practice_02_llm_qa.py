
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from common import constants
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

DATA_PATH = './data/qa_document.txt'
VECTOR_STORE_PATH = './vector_storage/local_vector_llm'

# 创建 Data Loader
loader = TextLoader(DATA_PATH, encoding='UTF-8')

# 创建 Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=60, chunk_overlap=10, separator='。')

# 创建 Embedding Model
embedding_model = OpenAIEmbeddings(openai_api_key=constants.API_KEY)

# 创建 Chat Model
chat = ChatOpenAI(openai_api_key=constants.API_KEY)

# 系统 Prompt Template
system_template = "你是一个人工智能助手，会根据提供的上下文信息给出准确的回答，如果提供的上下文中没有问题的答案，请说不知道。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# 用户 Prompt Template
human_template = "上下文信息：{content}，用户问题：{query}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 构建 Chat Template
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
print(chat_prompt)

# 构建 Chain
chain = LLMChain(llm=chat, prompt=chat_prompt)

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
    docs = db.similarity_search(query, k=1)
    print(docs)

    response = chain.run(content=docs, query=query)
    print(response)
