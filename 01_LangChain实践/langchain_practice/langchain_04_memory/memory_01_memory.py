import os
from common import constants
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

chat = ChatOpenAI(openai_api_key=constants.API_KEY)

# 创建 Memory
memory = ConversationBufferMemory(return_messages=True)

# 将 Memory 封装到 ConversationChain
conversation = ConversationChain(llm=chat, memory=memory)
print(conversation.prompt.template)

response = conversation.predict(input="请问中国的首都是哪里？")
print(response)

response = conversation.predict(input="这座城市有什么著名的旅游景点吗？")
print(response)
# {'input': '这里有什么特色小吃吗？',
# 'history': [HumanMessage(content='请问中国的首都是哪里？', additional_kwargs={}, example=False),
# AIMessage(content='中国的首都是北京。', additional_kwargs={}, example=False),
# HumanMessage(content='这里有什么著名的旅游景点吗？', additional_kwargs={}, example=False),
# AIMessage(content='是的，北京有很多著名的旅游景点。其中一些包括故宫、天安门广场、长城、颐和园、圆明园、天坛、鸟巢、水立方等。
# 这些景点都是中国历史和文化的重要象征，吸引了来自世界各地的游客。', additional_kwargs={}, example=False)]}

response = conversation.predict(input="这座城市有什么特色小吃吗？")
print(response)

response = conversation.predict(input="这座城市和上海有什么不同？")
print(response)