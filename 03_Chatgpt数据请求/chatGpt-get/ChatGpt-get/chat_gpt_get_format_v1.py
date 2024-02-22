#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from openai import OpenAI
import requests
import httpx
from common import constants


os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"


def text_davinci_003(input_content):
    client = OpenAI(
        base_url=constants.base_url,
        api_key=constants.API_KEY,
        http_client=httpx.Client(
            base_url=constants.base_url,
            follow_redirects=True,
        ),
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_content}
        ]
    )
    print(response)
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
    return choice_dict


def gpt_3_5_turbo(input_content):
    response_content = ''
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/70.0.3538.110 Safari/537.36',
        "Content-Type": "application/json",
        "Authorization": constants.AUTHORIZATION
    }
    data = {"model": "gpt-3.5-turbo",
            "messages": [{
                "role": "user",
                "content": input_content}]
            }
    try:
        start = time.time()
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers,
                                 json=data, timeout=100)
        print(response.text)
        if response.text:
            result_dict = json.loads(response.text)
            if 'choices' in result_dict and result_dict['choices']:
                choices_obj = result_dict['choices'][0]
                response_content = choices_obj.get('message', {}).get('content', '').strip()
                return response_content
        else:
            time.sleep(5)
        end = time.time()
        print('cost time: ', start - end)
    except Exception as e:
        print(e)
        time.sleep(10)
    return response_content


if __name__ == "__main__":
    res_content = text_davinci_003('世界上最大的淡水湖泊')
    print(res_content)
