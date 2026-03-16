import os
import json
import asyncio
import numpy as np
import pandas as pd
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# ==================== 配置 ====================
BASE_URL = "https://api2.aigcbest.top/v1"
API_KEY = "sk-YF5ia5p3v954rBNTJUlXQj000Bij8HsNnxzeiNihqe0bMJuF"
MODEL_NAME = "gpt-4.1"
DATA_ROOT = "data"
DOMAINS_FILE = os.path.join(DATA_ROOT, "phase1_domains_clustering.json")
WORKING_DIR_BASE = "./lightrag_domains"
SIMILARITY_THRESHOLD = 0.75

os.environ["OPENAI_API_KEY"] = API_KEY

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

# ==================== 读取数据 ====================
def load_domain_data(source_ids):
    """读取domain内所有source的表和列信息"""
    all_columns = []
    source_table_descs = {}  # source_id -> {table_name -> desc}

    for source_id in source_ids:
        source_dir = os.path.join(DATA_ROOT, source_id)
        if not os.path.exists(source_dir):
            continue
        source_table_descs[source_id] = {}

        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        for csv_name in csv_files:
            table_name = os.path.splitext(csv_name)[0]

            # 读样本
            try:
                df = pd.read_csv(os.path.join(source_dir, csv_name), nrows=5)
                col_types = df.dtypes.apply(lambda x: x.name).to_dict()
            except:
                df = pd.DataFrame()
                col_types = {}

            # 表描述
            table_view_path = os.path.join(source_dir, f"{table_name}_semantic_view.json")
            table_desc = "No description available."
            if os.path.exists(table_view_path):
                with open(table_view_path, 'r', encoding='utf-8') as f:
                    table_desc = json.load(f).get("description", "No description available.")
            source_table_descs[source_id][table_name] = table_desc

            # 列描述
            col_view_path = os.path.join(source_dir, f"{table_name}_columns_semantic_view.json")
            if not os.path.exists(col_view_path):
                continue
            with open(col_view_path, 'r', encoding='utf-8') as f:
                cols_semantic = json.load(f)

            for col_name, col_desc in cols_semantic.items():
                samples = []
                if not df.empty and col_name in df.columns:
                    samples = df[col_name].dropna().head(3).tolist()
                all_columns.append({
                    "id": f"{source_id}.{table_name}.{col_name}",
                    "source_id": source_id,
                    "table_name": table_name,
                    "column_name": col_name,
                    "description": col_desc,
                    "data_type": col_types.get(col_name, "unknown"),
                    "samples": samples,
                    "table_desc": table_desc
                })

    return all_columns, source_table_descs

# ==================== Step1: 骨架注入 ====================
async def inject_skeleton(rag, all_columns, source_table_descs):
    """直接注入结构关系：column→table→source，零LLM调用"""
    entities = []
    relationships = []
    seen_tables = set()
    seen_sources = set()

    for col in all_columns:
        source_id = col["source_id"]
        table_name = col["table_name"]
        table_id = f"{source_id}.{table_name}"

        # source节点
        if source_id not in seen_sources:
            entities.append({
                "entity_name": source_id,
                "entity_type": "SOURCE",
                "description": f"Data source {source_id}",
                "source_id": source_id
            })
            seen_sources.add(source_id)

        # table节点
        if table_id not in seen_tables:
            entities.append({
                "entity_name": table_id,
                "entity_type": "TABLE",
                "description": col["table_desc"],
                "source_id": source_id
            })
            relationships.append({
                "src_id": table_id,
                "tgt_id": source_id,
                "description": f"Table {table_name} belongs to data source {source_id}",
                "keywords": "belongs_to",
                "weight": 1.0,
                "source_id": source_id
            })
            seen_tables.add(table_id)

        # column节点
        entities.append({
            "entity_name": col["id"],
            "entity_type": "COLUMN",
            "description": f"{col['description']} Type: {col['data_type']}, Samples: {col['samples']}",
            "source_id": source_id
        })
        relationships.append({
            "src_id": col["id"],
            "tgt_id": table_id,
            "description": f"Column {col['column_name']} belongs to table {table_name}",
            "keywords": "belongs_to",
            "weight": 1.0,
            "source_id": source_id
        })

    await rag.ainsert_custom_kg({"entities": entities, "relationships": relationships})
    print(f"  ✅ 骨架注入完成：{len(entities)} 个实体，{len(relationships)} 条关系")

# ==================== Step2: 描述文本喂入（表级粒度）====================
async def insert_table_descriptions(rag, all_columns, source_table_descs):
    """按表级粒度拼自然语言描述，让LLM抽取语义关系"""
    # 按source+table分组
    from collections import defaultdict
    table_groups = defaultdict(list)
    for col in all_columns:
        key = (col["source_id"], col["table_name"])
        table_groups[key].append(col)

    docs = []
    for (source_id, table_name), cols in table_groups.items():
        col_descs = "\n".join([
            f"- {c['column_name']} ({c['data_type']}): {c['description']} Samples: {c['samples']}"
            for c in cols
        ])
        doc = (
            f"Table {source_id}.{table_name} is about: {cols[0]['table_desc']}\n"
            f"It contains the following columns:\n{col_descs}"
        )
        docs.append(doc)

    # 批量插入
    batch_size = 10
    for i in range(0, len(docs), batch_size):
        batch = "\n\n===\n\n".join(docs[i:i+batch_size])
        print(f"  插入表描述 {i+1}~{min(i+batch_size, len(docs))}/{len(docs)}...")
        await rag.ainsert(batch)
    print(f"  ✅ 表级描述插入完成：{len(docs)} 张表")

# ==================== Step3: Embedding相似关系注入 ====================
async def inject_embedding_similarities(rag, all_columns):
    """用sbert算列间相似度，高置信度的直接注入SIMILAR_TO关系"""
    print(f"  计算 {len(all_columns)} 列的embedding相似度...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{c['column_name']} {c['description']}" for c in all_columns]
    embeddings = model.encode(texts, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)

    relationships = []
    for i in range(len(all_columns)):
        for j in range(i+1, len(all_columns)):
            # 跳过同source
            if all_columns[i]['source_id'] == all_columns[j]['source_id']:
                continue
            score = float(sim_matrix[i][j])
            if score >= SIMILARITY_THRESHOLD:
                relationships.append({
                    "src_id": all_columns[i]['id'],
                    "tgt_id": all_columns[j]['id'],
                    "description": (
                        f"{all_columns[i]['id']} and {all_columns[j]['id']} "
                        f"are semantically similar columns (embedding similarity: {score:.3f}). "
                        f"Both appear to represent similar concepts: "
                        f"{all_columns[i]['description'][:100]}"
                    ),
                    "keywords": "similar_to, semantic_match",
                    "weight": score,
                    "source_id": all_columns[i]['source_id']
                })

    if relationships:
        await rag.ainsert_custom_kg({"entities": [], "relationships": relationships})
    print(f"  ✅ 相似关系注入完成：{len(relationships)} 条 SIMILAR_TO 边")
    return len(relationships)

# ==================== 主流程 ====================
async def main():
    with open(DOMAINS_FILE, 'r', encoding='utf-8') as f:
        domains = json.load(f)

    for domain_id, source_ids in domains.items():
        print(f"\n{'='*60}")
        print(f"🌐 处理 {domain_id}，数据源：{source_ids}")
        print(f"{'='*60}")

        if len(source_ids) < 2:
            print(f"⚠️  只有1个数据源，跳过")
            continue

        all_columns, source_table_descs = load_domain_data(source_ids)
        print(f"📄 共 {len(all_columns)} 列")

        rag = make_rag(domain_id)
        await rag.initialize_storages()

        print("\n[Step 1] 骨架注入...")
        await inject_skeleton(rag, all_columns, source_table_descs)

        print("\n[Step 2] 表级描述文本插入...")
        await insert_table_descriptions(rag, all_columns, source_table_descs)

        print("\n[Step 3] Embedding相似关系注入...")
        await inject_embedding_similarities(rag, all_columns)

        await rag.finalize_storages()
        print(f"\n✅ {domain_id} 建图完成")

if __name__ == "__main__":
    asyncio.run(main())