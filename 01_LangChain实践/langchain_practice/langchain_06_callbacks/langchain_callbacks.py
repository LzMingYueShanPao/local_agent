import os
from common import constants
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

"""
回调函数能获取日志记录，即大模型的思考过程
"""
handler = StdOutCallbackHandler()
llm = OpenAI(openai_api_key=constants.API_KEY)
prompt = PromptTemplate.from_template("1 + {number} = ")

print('---------------------------------')
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(number=2)
print('1 = ',response)

print('---------------------------------')
# 构造函数回调：首先，让我们在初始化链时显式设置StdOutCallbackHandler
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
response = chain.run(number=2)
print('2 = ', response)

print('---------------------------------')
# Use verbose flag: Then, let's use the `verbose` flag to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
response = chain.run(number=2)
print('3 = ', response)

print('---------------------------------')
# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(number=2, callbacks=[handler])
print(response)