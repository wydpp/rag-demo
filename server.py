import os.path
import time

import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from config.config import DATA_PATH
from service.rag_service_llama import llm, query_vector_store as llm_query, store_text_to_vector
from service.rag_service_native import query_vector_store as native_query, get_vector_store_index_native, native_llm

httpx.Timeout(30.0)

llama_collection_name = "rag-llama-demo"
native_collection_name = "rag-native-demo"

app = FastAPI()


@app.get("/init")
def init():
    # native加载data数据
    get_vector_store_index_native(native_collection_name)
    # llama加载data数据
    #store_text_to_vector(DATA_PATH, llama_collection_name)
    return {"message": "ok"}


@app.get("/llama/query")
def query(query: str):
    print(f"query: {query}")
    response = llm.complete(query)
    print(f"deepseek模型原生返回结果：{response}")
    print(type(response))
    return {"response": response.text}


@app.get("/llama/query-by-vector")
def query_by_vector(query: str):
    response = llm_query(llama_collection_name, query)
    print(f"使用向量数据库返回结果：{response}")
    return {"response": response}

@app.get("/native/query")
def query_by_vector(query: str):
    response = native_llm.get_completion_response(query)
    print(f"使用native返回结果：{response}")
    return {"response": response}

@app.get("/native/query-by-vector")
def query_by_vector(query: str):
    response = native_query(native_collection_name, query)
    print(f"使用native返回结果：{response}")
    return {"response": response}

