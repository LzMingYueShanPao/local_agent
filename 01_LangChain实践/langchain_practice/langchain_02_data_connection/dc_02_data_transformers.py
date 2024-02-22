from langchain.text_splitter import CharacterTextSplitter

with open("./data/data_transformer.txt", encoding='UTF-8') as f:
    open_ai_description = f.read()
"""
chunk_size：块大小
chunk_overlap：重叠部分大小
separator：分隔符
"""
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=120, chunk_overlap=0, separator='。'
)
"""
执行文本分割操作
"""
texts = text_splitter.split_text(open_ai_description)
print(texts)