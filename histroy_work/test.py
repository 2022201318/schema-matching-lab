import os
import uvicorn
from lightrag import LightRAG
from lightrag.api import app # 官方文档提到的 API 入口

# ================= 配置区 =================
# 指向你已经建好索引的目录
WORKING_DIR = "./lightrag_domains/domain_0" 

# 这里的初始化是为了让后端知道去哪里读 graphml 文件
rag = LightRAG(
    working_dir=WORKING_DIR,
    # 随便给个空的函数，因为我们主要是为了看图
    llm_model_func=lambda x: "", 
)

# 关键：将实例绑定到官方 API 服务的全局变量中
import lightrag.api.server as server
server.rag_obj = rag 

if __name__ == "__main__":
    print(f"🎨 正在加载 Domain: {WORKING_DIR}")
    print(f"🌐 访问地址: http://localhost:8020/webui/")
    # 启动官方内置的 FastAPI 服务
    uvicorn.run(app, host="0.0.0.0", port=8020)