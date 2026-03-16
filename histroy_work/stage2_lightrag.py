import os
import json
import asyncio
import pandas as pd
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# ==================== 配置 ====================
BASE_URL = "https://api2.aigcbest.top/v1"
API_KEY = "sk-YF5ia5p3v954rBNTJUlXQj000Bij8HsNnxzeiNihqe0bMJuF"
MODEL_NAME = "gpt-4.1"
DATA_ROOT = "data"
DOMAINS_FILE = os.path.join(DATA_ROOT, "phase1_domains_clustering.json")
OUTPUT_FILE = os.path.join(DATA_ROOT, "candidate_pairs.json")
WORKING_DIR_BASE = "./lightrag_domains"

os.environ["OPENAI_API_KEY"] = API_KEY

# ==================== LightRAG 初始化 ====================
async def custom_llm_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        MODEL_NAME, prompt, system_prompt=system_prompt,
        history_messages=history_messages, base_url=BASE_URL, api_key=API_KEY, **kwargs
    )

am_embedding = EmbeddingFunc(
    embedding_dim=1536,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts, model="text-embedding-3-small",
        base_url=BASE_URL, api_key=API_KEY
    )
)

def make_rag(domain_id):
    working_dir = os.path.join(WORKING_DIR_BASE, domain_id)
    os.makedirs(working_dir, exist_ok=True)
    return LightRAG(
        working_dir=working_dir,
        llm_model_func=custom_llm_complete,
        embedding_func=am_embedding
    )

# ==================== Step 1: 构建自然语言语义文档 ====================
def build_domain_docs(domain_id, source_ids):
    docs = []
    columns = []

    for source_id in source_ids:
        source_dir = os.path.join(DATA_ROOT, source_id)
        if not os.path.exists(source_dir):
            continue

        files = os.listdir(source_dir)
        csv_files = [f for f in files if f.endswith('.csv')]

        for csv_name in csv_files:
            table_name = os.path.splitext(csv_name)[0]
            csv_path = os.path.join(source_dir, csv_name)

            # 读取样本数据
            try:
                df = pd.read_csv(csv_path, nrows=5)
                col_types = df.dtypes.apply(lambda x: x.name).to_dict()
            except:
                df = pd.DataFrame()
                col_types = {}

            # 读取表级语义描述
            table_view_path = os.path.join(source_dir, f"{table_name}_semantic_view.json")
            table_desc = "No description available."
            if os.path.exists(table_view_path):
                with open(table_view_path, 'r', encoding='utf-8') as f:
                    table_desc = json.load(f).get("description", "No description available.")

            # --- 表级自然语言文档 ---
            col_names = df.columns.tolist() if not df.empty else []
            table_doc = (
                f"The table \"{table_name}\" belongs to data source {source_id}, "
                f"which is part of the {domain_id} business domain. "
                f"{table_desc} "
                f"This table contains the following columns: {', '.join(col_names)}. "
                f"Any column in this table can be referenced as {source_id}.{table_name}.<column_name>."
            )
            docs.append(table_doc)

            # 读取列级语义描述
            col_view_path = os.path.join(source_dir, f"{table_name}_columns_semantic_view.json")
            if not os.path.exists(col_view_path):
                continue

            with open(col_view_path, 'r', encoding='utf-8') as f:
                cols_semantic = json.load(f)

            # --- 列级自然语言文档 ---
            for col_name, col_desc in cols_semantic.items():
                samples = []
                if not df.empty and col_name in df.columns:
                    samples = df[col_name].dropna().head(3).tolist()
                data_type = col_types.get(col_name, "unknown")
                full_id = f"{source_id}.{table_name}.{col_name}"

                # 核心列文档：用自然语言描述这列是什么、属于哪里、长什么样
                col_doc = (
                    f"The column \"{col_name}\" is part of table \"{table_name}\" "
                    f"in data source {source_id} (domain: {domain_id}). "
                    f"Its full identifier is {full_id}. "
                    f"This column stores {col_desc} "
                    f"The data type is {data_type}, with representative sample values: {samples}. "
                    f"The parent table \"{table_name}\" is about: {table_desc[:200]}."
                )
                docs.append(col_doc)

                # 记录列元信息，供查询和解析使用
                columns.append({
                    "id": full_id,
                    "source_id": source_id,
                    "table_name": table_name,
                    "column_name": col_name,
                    "description": col_desc,
                    "data_type": data_type,
                    "samples": samples,
                    "table_desc": table_desc
                })

    return docs, columns


# ==================== Step 2: 构建LightRAG索引 ====================
async def build_index(rag, docs, domain_id):
    print(f"\n📦 [{domain_id}] 初始化存储引擎...")
    await rag.initialize_storages()

    print(f"🚀 [{domain_id}] 开始插入 {len(docs)} 条语义文档...")
    batch_size = 20
    for i in range(0, len(docs), batch_size):
        batch = "\n\n".join(docs[i:i + batch_size])
        print(f"  [{domain_id}] 插入第 {i+1}~{min(i+batch_size, len(docs))} 条...")
        await rag.ainsert(batch)

    print(f"✅ [{domain_id}] 索引构建完成")


# ==================== Step 3: 查询候选列对 ====================
async def query_candidate_pairs(rag, columns, domain_id):
    candidate_pairs = []
    seen = set()

    print(f"\n🔍 [{domain_id}] 开始查询候选列对，共 {len(columns)} 列...")

    for col in columns:
        query = (
            f"The column {col['id']} stores {col['description']} "
            f"Its data type is {col['data_type']} and sample values are {col['samples']}. "
            f"Based on semantic meaning, data type, and sample values, "
            f"which other columns in the knowledge base are semantically equivalent or similar to this column? "
            f"Please list their full identifiers in the format source_id.table_name.column_name, "
            f"and briefly explain why they are similar."
        )

        try:
            response = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        except Exception as e:
            print(f"  ⚠️  查询失败：{col['id']}，错误：{e}")
            continue

        # 解析回答：检查哪些列的完整id出现在response里
        for other_col in columns:
            if other_col['id'] == col['id']:
                continue
            if other_col['source_id'] == col['source_id']:
                continue
            # 用完整id匹配，避免误判
            if other_col['id'] in response:
                pair_key = tuple(sorted([col['id'], other_col['id']]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    candidate_pairs.append({
                        "col_a": col['id'],
                        "col_b": other_col['id'],
                        "domain": domain_id
                    })

    print(f"✅ [{domain_id}] 共生成 {len(candidate_pairs)} 个候选列对")
    return candidate_pairs


# ==================== 主流程 ====================
async def main():
    with open(DOMAINS_FILE, 'r', encoding='utf-8') as f:
        domains = json.load(f)

    all_candidate_pairs = []

    for domain_id, source_ids in domains.items():
        print(f"\n{'='*60}")
        print(f"🌐 处理 {domain_id}，包含数据源：{source_ids}")
        print(f"{'='*60}")

        if len(source_ids) < 2:
            print(f"⚠️  {domain_id} 只有1个数据源，跳过匹配")
            continue

        # Step 1: 构建自然语言语义文档
        docs, columns = build_domain_docs(domain_id, source_ids)
        print(f"📄 共构建 {len(docs)} 条语义文档，{len(columns)} 个列")

        if not columns:
            print(f"⚠️  {domain_id} 无有效列数据，跳过")
            continue

        # Step 2: 建索引
        rag = make_rag(domain_id)
        await build_index(rag, docs, domain_id)

        # Step 3: 查询候选对
        pairs = await query_candidate_pairs(rag, columns, domain_id)
        all_candidate_pairs.extend(pairs)

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_candidate_pairs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"🎯 全部完成！共生成 {len(all_candidate_pairs)} 个候选列对")
    print(f"📁 结果已保存至：{OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())