import os
from common import constants
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

chat = ChatOpenAI(openai_api_key=constants.API_KEY)

# 系统 Prompt Template
system_template = "以下是一段人类与人工智能之间的友好对话。 人工智能很健谈，并根据上下文提供了许多具体细节。 " \
           "如果人工智能不知道问题的答案，它就会如实说它不知道。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# 用户 Prompt Template
human_template = "{content}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 构建 Chat Template
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
print(chat_prompt)
# input_variables=['content'] messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
# template='以下是一段人类与人工智能之间的友好对话。 人工智能很健谈，并根据上下文提供了许多具体细节。
# 如果人工智能不知道问题的答案，它就会如实说它不知道。')),
# HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['content'], template='{content}'))]

chain = LLMChain(llm=chat, prompt=chat_prompt)
response = chain.run(content="我的宠物死了，能安慰安慰我吗？")
print(response)
# 我很抱歉听到关于你宠物的消息。失去宠物是一件非常痛苦的事情，他们是我们生活中的重要伙伴和家庭成员。
# 在这段艰难的时期，记得你并不孤单，周围还有很多人关心着你。试着回想一些美好的回忆，珍惜你和宠物在一起的时光，慢慢释放内心的悲伤。
# 如果你需要任何支持或愿意分享更多关于你宠物的故事，我都会倾听并陪伴在你身边。
