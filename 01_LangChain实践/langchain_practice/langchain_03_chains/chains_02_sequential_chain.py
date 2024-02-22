import os
from common import constants
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

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
# {'name': '日本偷袭珍珠港', 'introduction': '\n1941年12月7日，日本帝国海军突然对美国夏威夷群岛的珍珠港发动了突然袭击，这一事件被称为“珍珠港事件”。
# 这次偷袭造成了2400多名美国军人死亡，数十艘军舰和飞机被摧毁，给美国太平洋舰队造成了巨大的损失。这次偷袭引发了美国对日本的宣战，使得美国正式卷入第二次世界大战。
# 日本偷袭珍珠港的原因有多方面，包括日本对美国的经济封锁、美国对日本的军事干涉以及日本对太平洋地区的扩张野心等。这次事件的影响深远，不仅',
# 'review': '\n\n珍珠港事件是二战中最具有象征意义的事件之一，它不仅改变了美国的外交政策，也改变了世界的格局。这次突然袭击给美国太平洋舰队带来了巨大的损失，
# 也让美国人民感受到了战争的残酷和威胁。日本偷袭珍珠港的原因复杂多样，但最根本的原因还是日本对太平洋地区的扩张野心。日本帝国主义的侵略行为不仅给亚洲国家带来了灾难，
# 也给世界和平带来了严重威胁。\n\n作为历史评论家，我认为珍珠港事件的发生也暴露出了美国政府的一些问题。美国'}
print(result)