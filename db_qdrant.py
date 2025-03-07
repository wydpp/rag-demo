from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from qdrant_client.http.exceptions import UnexpectedResponse  # 捕获错误信息

from config import QDRANT_HOST, QDRANT_PORT

class Qdrant:

    def __init__(self):
        self.client = QdrantClient(host = QDRANT_HOST, port = QDRANT_PORT)
        self.size = 1536