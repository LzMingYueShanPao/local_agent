
import os
from common import constants
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

handler = StdOutCallbackHandler()
llm = OpenAI(openai_api_key=constants.API_KEY)
prompt = PromptTemplate.from_template("1 + {number} = ")

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(number=2)
print(response)

# Constructor callback: First, let's explicitly set the StdOutCallbackHandler when initializing our chain
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
response = chain.run(number=2)
print(response)

# Use verbose flag: Then, let's use the `verbose` flag to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
response = chain.run(number=2)
print(response)

# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(number=2, callbacks=[handler])
print(response)
