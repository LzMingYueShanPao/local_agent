
import os
from common import constants
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 创建 Chat Model
chat = ChatOpenAI(openai_api_key=constants.API_KEY)

# __call__: Messages in -> message out
# 通过向聊天模型传递一个或多个消息，您可以获取聊天完成的结果。响应将是一个消息。
messages = [HumanMessage(content="请将如下内容翻译成英文：今天天气真是不错")]
response = chat(messages)
print(response)
# content='The weather is really nice today.' additional_kwargs={} example=False

# SystemMessage 会作为前缀拼接
messages = [
    SystemMessage(content="请将如下内容翻译成英文："),
    HumanMessage(content="今天天气真是不错")
]
response = chat(messages)
print(response)
# content='The weather is really nice today.' additional_kwargs={} example=False

# generate: batch calls, richer outputs
# 您可以更进一步，使用generate为多个消息集生成完成结果。这将返回一个LLMResult，其中包含一个额外的消息参数
batch_messages = [
    [
        SystemMessage(content="请根据上下文回答以下问题："),
        HumanMessage(content="小明最喜欢吃烤鱼，最不喜欢吃西蓝花")
    ],
    [
        SystemMessage(content="请根据上下文回答以下问题："),
        HumanMessage(content="小明喜欢吃什么不喜欢吃什么？")
    ],
]
response = chat.generate(batch_messages)
print(response)
print(response.llm_output)
# generations=[[ChatGeneration(text='The weather is really nice today.',
# generation_info=None, message=AIMessage(content='The weather is really nice today.',
# additional_kwargs={}, example=False))], [ChatGeneration(text='Had a pancake for breakfast.',
# generation_info=None, message=AIMessage(content='Had a pancake for breakfast.', additional_kwargs={},
# example=False))]] llm_output={'token_usage': {'prompt_tokens': 76, 'completion_tokens': 14, 'total_tokens': 90},
# 'model_name': 'gpt-3.5-turbo'} run=RunInfo(run_id=UUID('3061c345-57b2-4893-8e7e-c8bf7a64471e'))
# {'token_usage': {'prompt_tokens': 76, 'completion_tokens': 14, 'total_tokens': 90}, 'model_name': 'gpt-3.5-turbo'}
