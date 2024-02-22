import os
from common import constants
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

# model_name = 'text-davinci-003'
model_name = 'gpt-3.5-turbo-instruct'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature, openai_api_key=constants.API_KEY)


# 创建Parser的目标格式
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    @validator('setup')
    def question_ends_with_question_mark(cls, field):
        if field[-1] != '?':
            raise ValueError("Badly formed question!")
        return field

# 创建输出Parser
parser = PydanticOutputParser(pydantic_object=Joke)
print(parser.get_format_instructions())
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#
# Here is the output schema:
# ```
# {"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"}, "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}}, "required": ["setup", "punchline"]}
# ```

# 创建Prompt
prompt = PromptTemplate(
    template="回答用户的问题\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

print(prompt)
# input_variables=['query'] partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to
# the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings",
# "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]}
# is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the
# output schema:\n```\n{"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"},
# "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}},
# "required": ["setup", "punchline"]}\n```'} template='回答用户的问题\n{format_instructions}\n{query}\n'

joke_query = "给我讲一个中文笑话"
_input = prompt.format_prompt(query=joke_query)
# text='回答用户的问题\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example,
# for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array",
# "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.
# The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n
# {"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"},
# "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}},
# "required": ["setup", "punchline"]}\n```\n给我讲一个中文笑话\n'
print(_input)

output = model(_input.to_string())
# {"setup": "为什么猪不能上树？", "punchline": "因为它们会被树枝挡住！"}
print(output)