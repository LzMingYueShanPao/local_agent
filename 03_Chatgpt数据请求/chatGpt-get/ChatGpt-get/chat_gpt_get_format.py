#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import openai
import requests
from common import constants


os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"


def open_ai_init():
    openai.api_key = constants.API_KEY


def text_davinci_003(input_content):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=input_content,
        temperature=0.9,
        max_tokens=200,
        stream=False,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
    # {
    #   "id": "cmpl-7ZqyKdg1pm6u0YTeI8XbjYbtcmeYS",
    #   "object": "text_completion",
    #   "created": 1688779460,
    #   "model": "text-davinci-003",
    #   "choices": [
    #     {
    #       "text": "\u662f\uff1a\n\n\u4e16\u754c\u4e0a\u6700\u5927\u7684\u6de1\
    #       u6c34\u6e56\u6cca\u662f\u4fc4\u7f57\u65af\u897f\u4f2f\u5229\u4e9a\u7684\u8d1d
    #       \u52a0\u5c14\u6e56\uff0c\u9762\u79ef\u7ea6\u4e3a37\u4e07\u5e73\u65b9\u516c\u91cc
    #       \uff0c\u662f\u4e16\u754c\u4e0a\u6700\u5927\u7684\u6de1\u6c34\u6e56\u6cca\u3002",
    #       "index": 0,
    #       "logprobs": null,
    #       "finish_reason": "stop"
    #     }
    #   ],
    #   "usage": {
    #     "prompt_tokens": 19,
    #     "completion_tokens": 101,
    #     "total_tokens": 120
    #   }
    # }
    res = response['choices'][0]['text']
    return res


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
        # {
        #   "id": "chatcmpl-7ZeuYFJwArOcjgUqmGQw4rb6GuRQf",
        #   "object": "chat.completion",
        #   "created": 1688733098,
        #   "model": "gpt-3.5-turbo-0613",
        #   "choices": [
        #     {
        #       "index": 0,
        #       "message": {
        #         "role": "assistant",
        #         "content": "根据最新统计数据，世界上最大的淡水湖泊是俄罗斯的贝加尔湖（Lake Baikal）。
        #         贝加尔湖位于西伯利亚地区，是世界上最深的湖泊之一，最大深度达到1,642米（5,387英尺）。
        #         它的面积约为31,722平方公里（12,248平方英里），相当于比利时的面积，使其成为全球第七大湖泊。
        #         贝加尔湖还是世界上最古老的湖泊之一，拥有许多独特的物种和生态系统。湖水的透明度极高，因此被称为“地球上的明亮之眼”。
        #         贝加尔湖也是一个受欢迎的旅游目的地，吸引着来自世界各地的游客。"
        #       },
        #       "finish_reason": "stop"
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 21,
        #     "completion_tokens": 256,
        #     "total_tokens": 277
        #   }
        # }
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
    open_ai_init()
    res_content = text_davinci_003('世界上最大的淡水湖泊')
    print(res_content)
