"""
文件读取和切分chunk
使用llama-index
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader

from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split_document(file_path: str) -> List[Document]:
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
    return text_splitter.split_documents(documents)


if __name__ == '__main__':
    file_path = 'D:/ai/code/rag-demo/data/粮食.txt'
    file_name = '粮食.txt'
    file_extension = 'txt'
    texts = load_and_split_document(file_path)
    print(f"切割为 {len(texts)} 个文本块")
    for text in texts:
        print(text)
    print(texts)
