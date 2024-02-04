import os
from common import constants
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

os.environ["OPENAI_API_KEY"] = constants.API_KEY

llm = OpenAI(temperature=0.5)
template = """
你是一位历史学家，请为以下历史事件写一个100字左右的介绍。
历史事件: {name}
历史学家: 以下为这个历史事件的介绍:"""
prompt_template = PromptTemplate(
    input_variables=["name"],
    template=template
)
introduction_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="introduction"
)

template = """
你是一位历史评论家，请你根据以下历史事件的介绍，写一个200字左右的历史评论。
历史事件介绍:
{introduction}
历史评论家对以上历史事件的评论:"""
prompt_template = PromptTemplate(
    input_variables=["introduction"],
    template=template
)
review_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="review"
)

overall_chain = SequentialChain(
    chains=[introduction_chain, review_chain],
    input_variables=["name"],
    output_variables=["introduction", "review"],
    verbose=True
)

result = overall_chain({
    "name": "日本偷袭珍珠港"
})
print(result)