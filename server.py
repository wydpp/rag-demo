import os.path
import time

import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from config.config import DATA_PATH
from service.rag_service import llm, query_vector_store, store_text_to_vector

app = FastAPI()

httpx.Timeout(30.0)

collection_name = "rag-demo"

# res = query_vector_store(collection_name, "你好!")
# print(f"初始化 向量数据库返回结果：{res}")


@app.get("/query")
def query(query: str):
    print(f"query: {query}")
    response = llm.complete(query)
    print(f"deepseek模型原生返回结果：{response}")
    print(type(response))
    return {"response": response.text}


@app.get("/query-by-vector")
def query_by_vector(query: str):
    response = query_vector_store(collection_name, query)
    print(f"使用向量数据库返回结果：{response}")
    return {"response": response}


@app.put("/upload-file")
async def update_file(file: UploadFile = File(...)):
    """
    文件上传接口

    - **file**: 上传的文件
    """
    try:
        # 读取文件内容
        contents = await file.read()
        print(f"文件名: {file.filename}")
        print(f"文件大小: {len(contents)} 字节")
        filename = str(int(time.time()))+file.filename
        file_path = os.path.join(DATA_PATH, filename)
        # 保存文件到本地
        with open(file_path, "wb") as f:
            f.write(contents)

        # 存储文件到向量数据库
        store_text_to_vector([file_path], collection_name)
        return JSONResponse(
            status_code=200,
            content={
                "message": "文件上传成功",
                "filename": file.filename,
                "size": len(contents)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"文件上传失败: {str(e)}"}
        )