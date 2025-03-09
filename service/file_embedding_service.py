"""
描述: 文件向量化的接口
"""
from abc import ABC
from typing import List, Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from config.config import EMBEDDING_MODEL_PATH, CHUNK_SIZE


class FileEmbedding:
    def __init__(self):
        self.client = pipeline(
            Tasks.sentence_embedding,
            model=EMBEDDING_MODEL_PATH,
            sequence_length=CHUNK_SIZE + 10  # sequence_length 代表最大文本长度，默认值为128
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的向量
        :param texts: 文本列表
        :return: 向量列表
        """
        input = {
            "source_sentence": texts
        }
        logger.info("start embedding  input:", input)
        res = self.client(input=input).get("text_embedding")
        logger.info("end embedding  input:", input)
        return res

    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的向量
        :param text: 文本
        :return: 向量
        """
        input = {
            "source_sentence": [text]
        }
        return self.client(input=input).get("text_embedding")[0]


# 全局实例化 FileEmbedding
file_embedding_instance = FileEmbedding()


# 实现 llama_index 的 BaseEmbedding 接口
class LlamaIndexEmbeddingAdapter(BaseEmbedding, ABC):
    file_embedding: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_embedding = file_embedding_instance

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.file_embedding.get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.file_embedding.get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.file_embedding.get_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self.file_embedding.get_embedding(query)


# 全局实例化 LlamaIndexEmbeddingAdapter
llama_index_embedding_adapter = LlamaIndexEmbeddingAdapter()

if __name__ == '__main__':
    # 测试代码
    file_embedding = file_embedding_instance
    texts = [
        "不可以，早晨喝牛奶不科学",
        "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
    ]
    # 测试 get_embeddings
    embeddings = file_embedding.get_embeddings(texts)
    print("Embeddings shape:", embeddings)
    print("Embeddings shape:", len(embeddings), len(embeddings[0]))  # 输出向量列表的长度和每个向量的维度

    # 测试 get_embedding
    embedding = file_embedding.get_embedding(texts[0])
    print("Single embedding shape:", len(embedding))  # 输出单个向量的维度
