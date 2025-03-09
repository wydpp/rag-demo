"""
qdrant向量数据库操作接口
"""
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config.config import QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE


# 向量数据库对象
class DBQdrantClient:

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT)
        self.size = VECTOR_SIZE

    # 获取集合
    def get_collection(self, collection_name):
        if self.client.collection_exists(collection_name=collection_name):
            return self.client.get_collection(collection_name=collection_name)
        else:
            # 不存在，创建集合
            return self.create_collection(collection_name=collection_name)

    # 创建集合
    def create_collection(self, collection_name):
        return self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.size,
                distance=Distance.COSINE
            )
        )

    def add_vectors(self, collection_name: str, points: List[PointStruct]):
        """
        添加向量数据
        :param collection_name: 集合名称
        :param points: 向量数据列表（每个元素是 PointStruct 对象）
        """
        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        print(f"Vectors added to collection '{collection_name}' successfully!")

    def search_vectors(self, collection_name, query_vector, limit=5):
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

if __name__ == '__main__':
    qdrant = DBQdrantClient()
    collection_name = "rag-demo-test"
    collection = qdrant.get_collection(collection_name=collection_name)
    print(collection)

    ## 添加向量
    points = [
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ]

    qdrant.add_vectors(collection_name, points)
    # 查询向量
    result = qdrant.search_vectors(collection_name, [0.2, 0.18, 0.22, 0.44],1)
    print(result)
