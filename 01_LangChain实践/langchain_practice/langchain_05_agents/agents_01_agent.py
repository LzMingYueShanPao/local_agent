import os
from common import constants
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

llm = OpenAI(openai_api_key=constants.API_KEY)

# 通过预设名称加载tools
# 加载数学文献库tools
tools = load_tools(["arxiv"])

# 初始化agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行agent
agent.run("介绍一下2203.02155这篇论文的创新点，使用中文进行回答?")
# > Entering new AgentExecutor chain...
#  I should use the arxiv tool to search for the paper and read its abstract
# Action: arxiv
# Action Input: 2203.02155
# Observation: Published: 2022-03-04
# Title: Training language models to follow instructions with human feedback
# Authors: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
# Summary: Making language models bigger does not inherently make them better at
# following a user's intent. For example, large language models can generate
# outputs that are untruthful, toxic, or simply not helpful to the user. In other
# words, these models are not aligned with their users. In this paper, we show an
# avenue for aligning language models with user intent on a wide range of tasks
# by fine-tuning with human feedback. Starting with a set of labeler-written
# prompts and prompts submitted through the OpenAI API, we collect a dataset of
# labeler demonstrations of the desired model behavior, which we use to fine-tune
# GPT-3 using supervised learning. We then collect a dataset of rankings of model
# outputs, which we use to further fine-tune this supervised model using
# reinforcement learning from human feedback. We call the resulting models
# InstructGPT. In human evaluations on our prompt distribution, outputs from the
# 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3,
# despite having 100x fewer parameters. Moreover, InstructGPT models show
# improvements in truthfulness and reductions in toxic output generation while
# having minimal performance regressions on public NLP datasets. Even though
# InstructGPT still makes simple mistakes, our results show that fine-tuning with
# human feedback is a promising direction for aligning language models with human
# intent.
# Thought: The paper concludes that fine-tuning language models with human feedback is a promising approach for improving their alignment with user intent and generating more helpful and accurate outputs.
# Thought: I now know the final answer
# Final Answer: The paper's main innovative point is the use of human feedback for fine-tuning language models to align them with user intent.