
import os
from langchain.prompts import PromptTemplate


os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 普通模板构建
prompt = PromptTemplate.from_template("请将下面内容翻译为英文 {content}：")
print(prompt)
prompt_format = prompt.format(content="明天又是美好的一天")
print(prompt_format)
# input_variables=['content'] output_parser=None partial_variables={} template='请将下面内容翻译为英文 {content}?'
# template_format='f-string' validate_template=True
