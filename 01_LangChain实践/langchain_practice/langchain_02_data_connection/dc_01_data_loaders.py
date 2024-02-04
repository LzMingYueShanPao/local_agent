
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./data/data.csv', encoding='UTF-8')
data = loader.load()
print(data)
