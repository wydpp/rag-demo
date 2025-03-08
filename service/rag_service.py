from typing import List

from llama_index.core import VectorStoreIndex, Settings, StorageContext, SimpleDirectoryReader
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config.config import QDRANT_HOST, QDRANT_PORT, DATA_PATH, DEEPSEEK_BASE_URL, DEFAULT_MODEL, DEEPSEEK_API_KEY
from service.file_embedding import llama_index_embedding_adapter

# 设置 deepseek api
llm = DeepSeek(model=DEFAULT_MODEL, api_key=DEEPSEEK_API_KEY, api_base=DEEPSEEK_BASE_URL)

Settings.llm = llm
Settings.embed_model = llama_index_embedding_adapter

# 初始化 Qdrant 客户端
qdrant_client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT)


def get_vector_store_index(collection_name: str):
    # 3. 检查集合是否存在
    collections = qdrant_client.get_collections().collections
    col = qdrant_client.get_collection(collection_name)
    collection_exists = any(col)
    # 4. 创建 Qdrant 向量存储
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    if collection_exists and col.points_count > 0:
        print("集合已存在，直接加载索引...")
        # 加载现有索引
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index
    else:
        print("集合不存在，创建集合并插入数据...")
        # 读取数据
        documents = SimpleDirectoryReader(DATA_PATH).load_data()
        for document in documents:
            print(f"document{document.text}")
        # 构建索引并插入数据
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print("索引构建完成")
        return index


# 文本向量化存储接口
def store_text_to_vector(file_path: List[str], collection_name: str):
    # 读取数据
    documents = SimpleDirectoryReader(input_files=file_path).load_data()
    index = get_vector_store_index(collection_name)
    for document in documents:
        index.insert(document)
        print(f"add document{document}")


def query_vector_store(collection_name: str, query: str, limit: int = 5):
    # 加载向量索引
    index = get_vector_store_index(collection_name)

    # 创建查询引擎
    query_engine = index.as_query_engine()

    # 执行查询
    response = query_engine.query(query)

    return response.response


if __name__ == '__main__':
    response = llm.complete("QwQ-32B是什么？")
    print(f"deepseek模型原生返回结果：{response}")

    # store_text_to_vector(["D:/ai/code/rag-demo/data/quq32b.txt"], "dpp-test")

    response = query_vector_store("dpp-test", "QwQ-32B是什么？")
    print(f"使用向量数据库返回结果：{response}")
