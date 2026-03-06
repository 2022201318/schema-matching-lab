import os
import json
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# 1. 沿用你跑通的配置
BASE_URL = "https://api2.aigcbest.top/v1" 
API_KEY = "sk-YF5ia5p3v954rBNTJUlXQj000Bij8HsNnxzeiNihqe0bMJuF"
MODEL_NAME = "gpt-4.1" 

os.environ["OPENAI_API_KEY"] = API_KEY

async def custom_llm_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        MODEL_NAME, prompt, system_prompt=system_prompt,
        history_messages=history_messages, base_url=BASE_URL, api_key=API_KEY, **kwargs
    )

am_embedding = EmbeddingFunc(
    embedding_dim=1536, 
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-small", # 保持使用你跑通的模型
        base_url=BASE_URL,
        api_key=API_KEY
    )
)

# 2. 初始化 LightRAG
WORKING_DIR = "./lightrag_db"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=custom_llm_complete,
    embedding_func=am_embedding
)

async def build_full_index():
    # A. 初始化
    print("📦 正在初始化存储引擎...")
    await rag.initialize_storages()

    # B. 读取之前合成的 693 条富语义数据
    input_file = "ready_for_cognee.json"
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到 {input_file}。请确保该文件在当前目录下。")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)
    
    print(f"📖 载入成功，共计 {len(enriched_data)} 条层级语义数据。")

    # C. 批量塞入 LightRAG
    # 提取所有 content 组成一个大的列表或分段插入
    print(f"🚀 开始构建全量血缘知识图谱...")
    
    # 建议做法：将所有 content 拼接成一个大文本或按条插入
    # 这里我们按条插入，以便于观察进度
    all_contents = [item["content"] for item in enriched_data if item.get("content")]
    
    # 实际上，LightRAG 的 ainsert 也可以接受长文本，
    # 为了保证实体关系提取最准确，我们分批喂进去
    batch_size = 20 # 每 20 条语义描述作为一个 chunk 插入
    for i in range(0, len(all_contents), batch_size):
        batch = "\n".join(all_contents[i : i + batch_size])
        print(f"正在处理第 {i} 到 {min(i + batch_size, len(all_contents))} 条数据...")
        await rag.ainsert(batch)

    print("✅ 全量索引构建完成！")

    # D. 深度推理测试
    print("\n🔍 正在基于 693 条血缘数据进行推理测试...")
    query = "请分析当前数据源中，哪些字段与‘用户消费行为’关联最紧密，并解释它们的血缘逻辑。"
    
    response = await rag.aquery(
        query, 
        param=QueryParam(mode="hybrid")
    )
    
    print("\n--- 深度分析结果 ---")
    print(response)

if __name__ == "__main__":
    asyncio.run(build_full_index())