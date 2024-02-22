import os
from common import constants
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

chat = ChatOpenAI(openai_api_key=constants.API_KEY)

# 创建 Memory
memory = ConversationBufferMemory(return_messages=True)

# 将 Memory 封装到 ConversationChain
conversation = ConversationChain(llm=chat, memory=memory)
# 0.  The following is a friendly conversation between a human and an AI.
# The AI is talkative and provides lots of specific details from its context.
# If the AI does not know the answer to a question, it truthfully says it does not know.
# Current conversation:
# {history}
# Human: {input}
# AI:
print('0. ',conversation.prompt.template)

response = conversation.predict(input="请问中国的首都是哪里？")
# 1.  中国的首都是北京。北京是中国的政治、文化和经济中心，也是中国最大的城市之一。北京市位于华北平原的北部，是一个拥有悠久历史和丰富文化遗产的城市。
print('1. ',response)

response = conversation.predict(input="这座城市有什么著名的旅游景点吗？")
# 2.  这座城市有许多著名的旅游景点，比如故宫、天坛、颐和园、长城等。故宫是中国明清两代的皇家宫殿，保存有大量珍贵的文物和艺术品。
# 天坛是一座古代祭祀皇天的建筑群，被认为是中国古代建筑的杰作。颐和园是一座皇家园林，有着美丽的湖泊和精致的建筑。长城是中国古代的防御工程，
# 被称为世界七大奇迹之一，是中国的象征之一。这些景点都吸引着大量游客前来观光游览。
print('2. ',response)

response = conversation.predict(input="这座城市有什么特色小吃吗？")
# 3.  这座城市有许多特色小吃，比如烤鸭、炸酱面、豆汁、驴打滚等。烤鸭是北京的特色美食之一，外脆里嫩，香味诱人。炸酱面是一道传统的面食，配上浓郁的酱料味道独特。
# 豆汁是一种古老的北京小吃，口感醇厚，喝起来清爽解腻。驴打滚则是一种传统的糕点，外脆内软，甜中带咸，回味无穷。这些小吃都是北京的饮食文化的重要组成部分，值得尝试。
print('3. ',response)

response = conversation.predict(input="这座城市和上海有什么不同？")
# 4.  这座城市和上海有很多不同之处。首先，北京是中国的政治中心，而上海是中国的经济中心。北京有许多古老的文化遗产和历史建筑，
# 而上海则是一个现代化的城市，拥有高楼大厦和繁华的商业街区。此外，北京的气候比较干燥，冬季寒冷，夏季炎热，而上海的气候湿润，四季分明。
# 总的来说，北京和上海各有各的特色，都是中国的重要城市。希望这些信息能帮助你更好地了解它们之间的区别。如果有其他问题，请随时问我。
print('4. ',response)