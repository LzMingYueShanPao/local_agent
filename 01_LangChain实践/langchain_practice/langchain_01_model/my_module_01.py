from common import constants
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

llm = ChatOpenAI(openai_api_key=constants.API_KEY)
llm.invoke("how can langsmith help with testing?")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
print(prompt)

chain = prompt | llm
print(chain)

chain.invoke({"input": "how can langsmith help with testing?"})

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

chain.invoke({"input": "how can langsmith help with testing?"})
print(chain)