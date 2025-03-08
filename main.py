# pip install fastapi uvicorn

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(
        app="server:app",  # 格式：文件名:应用实例名
        host="0.0.0.0",  # 允许所有IP访问
        port=8000,        # 端口号
        reload=True       # 开发模式：代码修改自动重载
    )