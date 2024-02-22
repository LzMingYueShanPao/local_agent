import os
from common import constants
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

llm = OpenAI(openai_api_key=constants.API_KEY)

# 通过预设名称加载tools
# SERP API 是一个提供搜索引擎结果页面数据的服务。通过使用 SERP API，你可以获取各种搜索引擎（如谷歌、百度、雅虎等）返回的搜索结果、广告数据、地图数据等。
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=constants.SERP_API_KEY)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("这几天北京的最高温度是多少度？这个数字的2次方是多少？")
# > Entering new AgentExecutor chain...
#  I should use the calculator to find the square of the highest temperature in Beijing.
# Action: Calculator
# Action Input: 36
# Observation: Answer: 36
# Thought: Now I have to multiply 36 by itself to find the square.
# Action: Calculator
# Action Input: 36*36
# Observation: Answer: 1296
# Thought: I now know the final answer.
# Final Answer: The square of the highest temperature in Beijing in the past few days is 1296 degrees.