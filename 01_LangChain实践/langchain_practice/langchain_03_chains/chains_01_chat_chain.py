
import os
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
# input_variables=['content'] output_parser=None partial_variables={}
# messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
# output_parser=None, partial_variables={}, template='以下是一段人类与人工智能之间的友好对话。
# 人工智能很健谈，并根据上下文提供了许多具体细节。 如果人工智能不知道问题的答案，它就会如实说它不知道。',
# template_format='f-string', validate_template=True), additional_kwargs={}),
# HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['content'],
# output_parser=None, partial_variables={}, template='{content}', template_format='f-string',
# validate_template=True), additional_kwargs={})]

chain = LLMChain(llm=chat, prompt=chat_prompt)
response = chain.run(content="我的宠物死了，能安慰安慰我吗？")
print(response)
# 我很抱歉听到这个消息。失去宠物是一件非常难过的事情，我能理解你现在的感受。宠物是我们生活中的一部分，
# 它们陪伴我们度过了很多时光，我们会因为它们的离开而感到孤独和悲伤。我希望你能够坚强地面对这个事实，
# 同时也要记得珍惜你们一起度过的美好时光，这些回忆将永远留在你的心中。
