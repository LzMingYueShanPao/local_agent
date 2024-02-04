
import os
from common import constants
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

llm = OpenAI(openai_api_key=constants.API_KEY)

# 通过预设名称加载tools
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=constants.SERP_API_KEY)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("这几天北京的最高温度是多少度？这个数字的2次方是多少？")
# > Entering new  chain...
#  我应该先查一下北京的当前温度，然后再用计算器来计算2次方。
# Action: Search
# Action Input: "Beijing Temperature"
# Observation: Beijing, Beijing. 86°F.
# Thought: 我现在可以用计算器计算2次方了
# Action: Calculator
# Action Input: 86^2
# Observation: Answer: 7396
# Thought: 我现在知道最终的答案了
# Final Answer: 北京的最高温度是86°F，其2次方是7396。
