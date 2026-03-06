import os
import json
import asyncio
import cognee
from cognee.api.v1.visualize import visualize_graph

# --- 1. 严格按照文档要求的 Custom Provider 配置 ---
os.environ["LLM_PROVIDER"] = "custom"
os.environ["LLM_ENDPOINT"] = "https://api.aigcbest.com/v1"
os.environ["LLM_API_KEY"] = "sk-JoM1NhxV2GRfAFaokDopuG6i5YFqwBh4j0gadLlPRZNHUCkC"

# 关键：当 PROVIDER="custom" 时，模型名前必须加 openai/ 前缀
os.environ["LLM_MODEL"] = "openai/qwen2.5-72b-instruct" 

# 额外保险：文档提到如果不配 Embedding，它会默认找 OpenAI
# 我们让 Embedding 也复用中转站配置
os.environ["EMBEDDING_PROVIDER"] = "custom"
os.environ["EMBEDDING_ENDPOINT"] = "https://api.aigcbest.com/v1"
os.environ["EMBEDDING_API_KEY"] = "sk-JoM1NhxV2GRfAFaokDopuG6i5YFqwBh4j0gadLlPRZNHUCkC"
os.environ["EMBEDDING_MODEL"] = "openai/text-embedding-3-small"

async def main():
    # 2. 加载数据
    try:
        with open("ready_for_cognee.json", "r", encoding='utf-8') as f:
            rich_data = json.load(f)
    except Exception as e:
        print(f"❌ 加载 JSON 失败: {e}")
        return

    print(f"✅ 已配置 Custom Provider，准备录入 {len(rich_data)} 条数据...")

    # 3. 录入数据
    for i, entry in enumerate(rich_data):
        meta = entry["metadata"]
        tagged_content = f"[{meta['type'].upper()}: {meta.get('source_id')}] {entry['content']}"
        
        try:
            await cognee.add(data = tagged_content, dataset_name = "experiment_final")
        except Exception as e:
            print(f"❌ 录入中断 (第 {i} 条): {e}")
            print("💡 提示：如果仍报 Key 错误，请在终端尝试: export LLM_PROVIDER=custom")
            return

        if (i + 1) % 100 == 0:
            print(f"已录入 {i+1} 条...")

    # 4. 构建图谱
    print("🧠 正在构建图谱 (Cognify)...")
    await cognee.cognify()
    
    # 5. 生成可视化
    print("🎨 正在生成图谱可视化 HTML...")
    await visualize_graph(os.path.join(os.getcwd(), "schema_graph.html"))
    print("🎉 任务完成！请查看 schema_graph.html")

if __name__ == "__main__":
    asyncio.run(main())