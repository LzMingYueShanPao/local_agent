import os
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

# Chat Model 模板构建
# 系统 Prompt
template = "请将下面内容从 {input_language} 翻译为 {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 用户 Prompt
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 创建 Chat Prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt.format_messages(input_language="中文", output_language="英文", text="希望明天能收到好消息")
# input_variables=['input_language', 'output_language', 'text']
# messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input_language',
# 'output_language'], template='请将下面内容从 {input_language} 翻译为 {output_language}.')),
# HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['text'], template='{text}'))]
print(chat_prompt)