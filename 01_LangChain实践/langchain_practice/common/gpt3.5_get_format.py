# pip install openai==1.9.0

from openai import OpenAI
import httpx
import json
import os
import common.constants as constants

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

client = OpenAI(
    base_url=constants.OPENAI_URL,
    api_key=constants.API_KEY,
    http_client=httpx.Client(
        base_url=constants.OPENAI_URL,
        follow_redirects=True,
    ),
)

def gpt_3_5_turbo(input_content):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "你是个乐于助人的助手。"},
        {"role": "user", "content": input_content}
      ]
    )
    choice = response.choices[0]
    # 将Choice对象转换为字典
    choice_dict = {
        "finish_reason": choice.finish_reason,
        "index": choice.index,
        "logprobs": choice.logprobs,
        "message": {
            "content": choice.message.content,
            "role": choice.message.role,
            "function_call": choice.message.function_call,
            "tool_calls": choice.message.tool_calls
        }
    }
    message = choice_dict['message']
    return message

if __name__ == '__main__':
    message = gpt_3_5_turbo("说说泰国的旅游业")
    print(message)
