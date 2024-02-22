import openai
import os
from common import constants

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

openai.api_key = constants.API_KEY
model_engine = "gpt-3.5-turbo"