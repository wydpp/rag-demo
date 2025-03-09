"""
文件读取和切分chunk
使用llama-index
"""
from loguru import logger
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader

from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_file_and_split_document(file_path: str) -> List[Document]:
    """
    切分document
    :param file_path:
    :return:
    """
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    # 文本
    texts = [doc.text for doc in documents]
    # 元数据
    metadatas = [doc.metadata for doc in documents]
    # 切割后的文档
    logger.info(f"切割前文档id={documents[0].doc_id}")
    return text_splitter.create_documents(texts, metadatas)


def load_filedir_and_split_document(file_directory: str) -> List[Document]:
    """
    切分document
    :param file_path:
    :return:
    """
    documents = SimpleDirectoryReader(input_dir=file_directory, required_exts=['.txt']).load_data()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    # 文本
    texts = [doc.text for doc in documents]
    # 元数据
    metadatas = [doc.metadata for doc in documents]
    # 切割后的文档
    return text_splitter.create_documents(texts, metadatas)


if __name__ == '__main__':
    file_path = 'D:/ai/code/rag-demo/data'
    texts = load_filedir_and_split_document(file_path)
    print(f"切割为 {len(texts)} 个文本块")
    print(texts[0])
    print(texts[1])
