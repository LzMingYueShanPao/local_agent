import os
from common import constants
from langchain.llms import OpenAI

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

llm = OpenAI(openai_api_key=constants.API_KEY)

# __call__: string in -> string out
# 使用LLM最简单的方式是通过调用：传入一个字符串，获取一个字符串结果。
single_response = llm("讲一个笑话吧")
print(single_response)
# 一只熊跟一只狐狸正在河边聊天，熊问狐狸：“你有没有船？”狐狸说：“没有，你有吗？”
# 熊说：“没有，那我们怎么渡河呢？”狐狸回答：“渡什么河？我们现在就站在河对岸！”

# generate: batch calls, richer outputs
# generate允许您使用字符串列表调用模型，得到比单文本更完整的响应。这个完整的响应可以包括多个top response和其他特定于LLM提供商的信息。
llm_result = llm.generate(["讲一个笑话吧", "写一首关于爱情的诗"])
print(llm_result)
print(llm_result.generations)
print(llm_result.llm_output)
# generations=[[Generation(text='\n\n两个猴子在玩石头剪刀布，一个猴子说：“你先出拳！”另一只猴子说：“怎么出？”第一只猴子说：“你拿石头！”',
# generation_info={'finish_reason': 'stop', 'logprobs': None})], [Generation(text='\n\n爱情是一朵芬芳的花，
# \n它真实无私的慷慨\n在一个深深的拥抱里，\n它让你无比心安\n\n它给你一个美丽的梦，\n它让你活得更有力\n它让你去追寻梦想，
# \n把梦想变成真\n\n爱情使你无比强大，\n它让你的心更坚定\n它让你在黑暗里，\n发现一片希望\n\n爱情是一份温柔的',
# generation_info={'finish_reason': 'length', 'logprobs': None})]]
# llm_output={'token_usage': {'completion_tokens': 371, 'prompt_tokens': 31, 'total_tokens': 402},
# 'model_name': 'text-davinci-003'} run=RunInfo(run_id=UUID('ecdb2595-1f1e-4241-a37c-799c030b2db1'))
