import os
from langchain.prompts import PromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

# 普通模板构建
prompt = PromptTemplate.from_template("请将下面内容翻译为英文 {content}：")
print(prompt)
prompt_format = prompt.format(content="明天又是美好的一天")
print(prompt_format)
# input_variables=['content'] template='请将下面内容翻译为英文 {content}：'
# 请将下面内容翻译为英文 明天又是美好的一天：
