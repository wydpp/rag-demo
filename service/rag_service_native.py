"""
原生实现
"""
import uuid
from typing import List

from langchain_core.documents import Document
from loguru import logger
from qdrant_client.http.models import PointStruct

from client.db_qdrant_client import DBQdrantClient
from client.llm_deepseek_client import LLMDeepseekClient
from config.config import DATA_PATH
from service.file_embedding_service import file_embedding_instance
from utils.file_util import load_filedir_and_split_document

# 设置 deepseek api
native_llm = LLMDeepseekClient()

# 初始化 Qdrant 客户端
qdrant_client = DBQdrantClient()


def get_vector_store_index_native(collection_name: str):
    """
    获取向量数据库集合对象
    :param collection_name:
    :return:
    """
    # 3. 检查集合是否存在
    collection = qdrant_client.get_collection(collection_name)
    # 4. 创建 Qdrant 向量存储
    if collection.points_count > 0:
        print("集合有数据，直接返回")
    else:
        print("集合没有数据，初始化插入数据...")
        # 读取数据
        documents = load_filedir_and_split_document(DATA_PATH)
        logger.info(f"读取数据完成，共{len(documents)}条数据")
        # 构建索引并插入数据
        file_to_vector(documents, collection_name)
        print("数据初始化完成")
    return collection


def file_to_vector(documents: List[Document], collection_name: str):
    """
    文本转换成向量并存储（分批次插入）
    :param documents: 文档列表
    :param collection_name: 向量数据库集合名称
    :return:
    """
    # 分批次处理，每个向量维度是512，数据太大会导致占用内存过大
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        docs = documents[i:i + batch_size]  # 取当前批次的文本
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        # **批量计算 embedding，而不是一次性计算所有**
        batch_embeddings = file_embedding_instance.get_embeddings(texts)

        # 构建 payloads（存入向量数据库的文本内容）
        payloads = build_payloads(texts, metadatas)

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
            for embedding, payload in zip(batch_embeddings, payloads)
        ]
        logger.info(f"开始存入 {len(points)} 条数据")
        qdrant_client.add_vectors(collection_name, points)
        logger.info(f"已存入 {len(points)} 条数据")

    logger.info(f"所有文本数据已存入向量数据库，共 {len(documents)} 条数据")


def build_payloads(texts, metadatas):
    payloads = [
        {

            "page_content": text,
            "metadata": metadata,
        }
        for text, metadata in zip(texts, metadatas)
    ]
    return payloads


def query_vector_store(collection_name: str, query: str, limit: int = 5):
    """
    1.先把query转成向量
    2.从向量数据库查询数据
    3.根据query和向量查询结果构建promotion
    4.调用llm接口查询
    :param collection_name:
    :param query:
    :param limit:
    :return:
    """
    # 1.查询内容转向量
    query_vector = file_embedding_instance.get_embedding(query)
    # 2.从向量数据库查询数据
    results = qdrant_client.search_vectors(collection_name, query_vector, limit)
    print(results)
    if not (results and results.points):
        return "没有查询到相关数据"
    # 从向量数据库查询到的相关数据
    relation_content = ""
    for result in results.points:
        relation_content += result.payload['page_content'] + "\n"
    # 3.根据query和向量查询结果构建promotion
    prompt = f"""
    你是一位 AI 助手，负责回答用户问题。请根据提供的检索原始内容，优化回答，使其更加完整、准确和可读。
    
    **用户问题**: {query}
    
    **检索到的原始内容**:
    {relation_content}
    
    **通过上面检索到的原始内容，进行回答优化后的回答**:
    """
    print(f"构造的prompt==> {prompt}")

    # 4.调用llm接口查询
    res = native_llm.get_completion_response(prompt)

    return res


if __name__ == '__main__':
    response = native_llm.complete("QwQ-32B是什么？")
    print(f"deepseek模型原生返回结果：{response}")
