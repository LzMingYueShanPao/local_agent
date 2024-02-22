from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./data/data.csv', encoding='UTF-8')
data = loader.load()
# [Document(page_content='index: data_01\nen: spring\nzh: 春天', metadata={'source': './data/data.csv', 'row': 0}),
# Document(page_content='index: data_02\nen: summer\nzh: 夏天', metadata={'source': './data/data.csv', 'row': 1}),
# Document(page_content='index: data_03\nen: autumn\nzh: 秋天', metadata={'source': './data/data.csv', 'row': 2}),
# Document(page_content='index: data_04\nen: winter\nzh: 冬天', metadata={'source': './data/data.csv', 'row': 3})]
print(data)
for document in data:
    result = {}
    lines = document.page_content.split('\n')
    for line in lines:
        key, value = line.split(': ')
        result[key] = value
    print(result)
# {'index': 'data_01', 'en': 'spring', 'zh': '春天'}
# {'index': 'data_02', 'en': 'summer', 'zh': '夏天'}
# {'index': 'data_03', 'en': 'autumn', 'zh': '秋天'}
# {'index': 'data_04', 'en': 'winter', 'zh': '冬天'}
