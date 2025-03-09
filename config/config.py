import os

from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 获取 API 密钥
DEEPSEEK_API_KEY = os.environ["DASHSCOPE_API_KEY"]
# 使用阿里云百炼的免费模型
#DEEPSEEK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 本地部署的deepseek大模型
DEEPSEEK_BASE_URL = "http://127.0.0.1:8080/v1"

# 向量化模型地址
#EMBEDDING_MODEL_PATH = "D:/ai/model/iic/nlp_gte_sentence-embedding_chinese-large"
EMBEDDING_MODEL_PATH = "D:/ai/model/iic/nlp_gte_sentence-embedding_chinese-small"

MODELS = [
    "deepseek-r1",  # 免费的deepseek1.5b模型
]

DEFAULT_MODEL = MODELS[0]

# 模型支持的最大输入token数量
MODEL_MAX_TOKENS = {
    "deepseek-r1": 32768,
}
# 向量数据库地址
QDRANT_HOST = "http://192.168.0.111"
# 向量数据库端口
QDRANT_PORT = 6333

VECTOR_SIZE = 512

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEFAULT_MAX_TOKENS = 32,768

DATA_PATH = "D:/ai/code/rag-demo/data"