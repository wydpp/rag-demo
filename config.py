import os

from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 获取 API 密钥
DEEPSEEK_API_KEY = os.environ["DASHSCOPE_API_KEY"]
# 使用阿里云百炼的免费模型
DEEPSEEK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 阿里云模型地址 https://help.aliyun.com/zh/model-studio/getting-started/models
MODELS = [
    "deepseek-r1-distill-qwen-1.5b",  # 免费的deepseek1.5b模型
]

DEFAULT_MODEL = MODELS[0]

# 模型支持的最大输入token数量
MODEL_MAX_TOKENS = {
    "deepseek-r1-distill-qwen-1.5b": 32768
}
# 向量数据库地址
QDRANT_HOST = "localhost"
# 向量数据库端口
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEFAULT_MAX_TOKENS = 2000