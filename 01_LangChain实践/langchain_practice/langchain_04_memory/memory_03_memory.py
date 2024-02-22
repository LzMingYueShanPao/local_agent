import os
from common import constants
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

chat = ChatOpenAI(openai_api_key=constants.API_KEY)

# 创建 ConversationSummaryMemory
# 在对话开始后对对话进行总结，将摘要保存在memory中
memory = ConversationSummaryMemory(llm=chat)

# 将 Memory 封装到 ConversationChain
conversation = ConversationChain(llm=chat, memory=memory)

response = conversation.predict(input="请问中国的首都是哪里？")
print(response)

response = conversation.predict(input="这里有什么著名的旅游景点吗？")
print(response)

response = conversation.predict(input="这里有什么特色小吃吗？")
print(response)

response = conversation.predict(input="这里和上海有什么不同？")
print(response)