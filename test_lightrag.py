import os
import asyncio
import numpy as np
import json
import re
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# --- 1. 配置参数 ---
BASE_URL = "https://api2.aigcbest.top/v1" 

MODEL_NAME = "gpt-4.1" 
WORKING_DIR = "./lightrag_stratified_final"

# --- 2. 辅助函数：鲁棒的 JSON 提取器 ---
def extract_json_from_text(text):
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        return json.loads(match.group(1)) if match else json.loads(text)
    except: return None

# 新增：审计记录函数，确保中间过程实时存盘
def save_audit_log(step_name, question, answer):
    filename = f"AUDIT_{step_name}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"--- QUESTION ---\n{question}\n\n--- AI RESPONSE ---\n{answer}\n")
    print(f"📄 审计记录已实时保存至: {filename}")

# --- 3. 定义适配器 ---
async def my_llm_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        MODEL_NAME, prompt, system_prompt=system_prompt,
        history_messages=history_messages, base_url=BASE_URL,
        api_key=API_KEY, **kwargs
    )

async def my_embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts, model="text-embedding-3-small", base_url=BASE_URL, api_key=API_KEY
    )

# --- 4. 初始化 ---
if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=my_llm_complete,
    embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=my_embedding_func),
    llm_model_max_async=4,
    embedding_func_max_async=16
)

async def run_stratified_analysis():
    await rag.initialize_storages()

    # ==========================================
    # PHASE 1: 导入源并识别所有业务域
    # ==========================================
    print("\n🚀 [PHASE 1] Indexing Sources & Clustering Domains...")
    with open("KB_LEVEL_1_SOURCES.txt", "r", encoding='utf-8') as f:
        await rag.ainsert(f.read())
    
    q1 = (
        "TASK: Analyze all indexed Data Sources.\n"
        "1. Categorize them into 5-8 distinct Business Domains.\n"
        "2. POLICY: '宁滥勿缺' (Rather over-cluster than leave orphans). If uncertain, group them.\n"
        "OUTPUT_FORMAT: Provide a strict JSON ONLY: "
        "{'domains': [{'name': '...', 'sources': [...], 'scope': '...'}]}"
    )
    r1 = await rag.aquery(q1, param=QueryParam(mode="global"))
    
    # --- 实时输出并写文件 ---
    print(f"\n[PHASE 1 原文输出]:\n{r1}")
    save_audit_log("STEP1_DOMAINS", q1, r1)
    
    domains_data = extract_json_from_text(r1)
    if not domains_data:
        print(f"❌ Error: JSON 提取失败。")
        return

    print(f"✅ Identified {len(domains_data['domains'])} domains.")

    # ==========================================
    # PHASE 2: 表级关联
    # ==========================================
    print("\n🚀 [PHASE 2] Indexing Tables & Linking Per Domain...")
    with open("KB_LEVEL_2_TABLES.txt", "r", encoding='utf-8') as f:
        await rag.ainsert(f.read())

    domain_links = []
    for domain in domains_data['domains']:
        d_name, d_sources = domain['name'], domain['sources']
        print(f"\n🔍 Analyzing Domain: {d_name}...")

        q2 = (
            f"FOCUS: Domain '{d_name}' (Sources: {d_sources}).\n"
            f"TASK: Find ALL potential table pairs with semantic overlaps.\n"
            f"POLICY: '宁滥勿缺' (Recall over Precision). If uncertain, assume they are linked.\n"
            f"OUTPUT: A structured Markdown table."
        )
        r2 = await rag.aquery(q2, param=QueryParam(mode="global"))
        
        # --- 实时输出并写文件 ---
        print(f"[PHASE 2 {d_name} 结果]:\n{r2}")
        safe_name = d_name.replace("/", "_").replace(" ", "_")
        save_audit_log(f"STEP2_TABLES_{safe_name}", q2, r2)
        
        domain_links.append({"name": d_name, "links": r2})

    # ==========================================
    # PHASE 3: 列级匹配
    # ==========================================
    print("\n🚀 [PHASE 3] Indexing Column Details & Final Mapping...")
    with open("KB_LEVEL_3_COLUMNS.txt", "r", encoding='utf-8') as f:
        await rag.ainsert(f.read())

    for item in domain_links:
        d_name, l_ctx = item['name'], item['links']
        print(f"\n🎯 Mapping Columns for: {d_name}...")

        q3 = (
            f"DOMAIN: {d_name}\nSUSPECTED LINKS:\n{l_ctx}\n\n"
            f"TASK: Deep-dive column mapping. If a match is possible, RETAIN IT.\n"
            f"POLICY: 'Aggressive Preservation' (宁滥勿缺). Do not filter out low-confidence matches.\n"
            f"OUTPUT: [TableA.Col] <-> [TableB.Col] | Confidence | Logic."
        )
        r3 = await rag.aquery(q3, param=QueryParam(mode="hybrid", top_k=60))
        
        # --- 实时输出并写文件 ---
        print(f"[PHASE 3 {d_name} 结果]:\n{r3}")
        safe_name = d_name.replace("/", "_").replace(" ", "_")
        save_audit_log(f"STEP3_COLUMNS_{safe_name}", q3, r3)

    await rag.finalize_storages()
    print("\n✨ All Phases Complete.")

if __name__ == "__main__":
    asyncio.run(run_stratified_analysis())