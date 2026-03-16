import os
import json
import asyncio
from itertools import combinations
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
WORKING_DIR_BASE = "./lightrag_domains"
OUTPUT_FILE = os.path.join(DATA_ROOT, "final_matches.json")

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

# ==================== 加载列信息 ====================
def load_source_columns(source_ids):
    source_columns = {}
    for source_id in source_ids:
        source_dir = os.path.join(DATA_ROOT, source_id)
        if not os.path.exists(source_dir):
            continue
        source_columns[source_id] = []
        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        for csv_name in csv_files:
            table_name = os.path.splitext(csv_name)[0]
            col_view_path = os.path.join(source_dir, f"{table_name}_columns_semantic_view.json")
            if os.path.exists(col_view_path):
                with open(col_view_path, 'r', encoding='utf-8') as f:
                    for col_name in json.load(f).keys():
                        source_columns[source_id].append(
                            f"{source_id}.{table_name}.{col_name}"
                        )
    return source_columns

# ==================== 单对source查询 ====================
async def query_source_pair(rag, source_a, source_b, cols_a, cols_b, domain_id):
    cols_a_str = "\n".join([f"  - {c}" for c in cols_a])
    cols_b_str = "\n".join([f"  - {c}" for c in cols_b])

    query = (
        f"Based on the knowledge graph, determine which columns from {source_a} "
        f"and {source_b} are semantically equivalent and should be matched.\n\n"
        f"Columns in {source_a}:\n{cols_a_str}\n\n"
        f"Columns in {source_b}:\n{cols_b_str}\n\n"
        f"Only list column pairs that you are confident represent the same real-world concept."
    )

    try:
        response = await rag.aquery(
            query,
            param=QueryParam(
                mode="hybrid",
                response_type="Bullet Points",
                user_prompt=(
                    "Return ONLY confirmed matching column pairs, one per line:\n"
                    "source_id.table_name.column_name <-> source_id.table_name.column_name\n"
                    "Be conservative — only include pairs you are certain match.\n"
                    "No explanations, no uncertain pairs."
                )
            )
        )
    except Exception as e:
        print(f"  ⚠️  查询失败 ({source_a} vs {source_b}): {e}")
        return []

    # 解析结果
    pairs = []
    seen = set()
    all_cols = set(cols_a + cols_b)

    for line in response.strip().split('\n'):
        if '<->' not in line:
            continue
        parts = line.strip().split('<->')
        if len(parts) != 2:
            continue
        col_a = parts[0].strip().strip('-').strip()
        col_b = parts[1].strip()

        if col_a not in all_cols or col_b not in all_cols:
            continue
        if col_a.split('.')[0] == col_b.split('.')[0]:
            continue

        pair_key = tuple(sorted([col_a, col_b]))
        if pair_key not in seen:
            seen.add(pair_key)
            pairs.append({
                "col_a": col_a,
                "col_b": col_b,
                "domain": domain_id
            })

    return pairs

# ==================== 主流程 ====================
async def main():
    with open(DOMAINS_FILE, 'r', encoding='utf-8') as f:
        domains = json.load(f)

    all_matches = []

    for domain_id, source_ids in domains.items():
        print(f"\n{'='*60}")
        print(f"🌐 处理 {domain_id}，数据源：{source_ids}")
        print(f"{'='*60}")

        # 加载列信息
        source_columns = load_source_columns(source_ids)
        valid_sources = [s for s in source_ids if source_columns.get(s)]

        if len(valid_sources) < 2:
            print(f"⚠️  有效数据源不足2个，跳过")
            continue

        print(f"📄 有效数据源：{valid_sources}")
        for s in valid_sources:
            print(f"  {s}: {len(source_columns[s])} 列")

        # 检查索引目录是否存在
        working_dir = os.path.join(WORKING_DIR_BASE, domain_id)
        if not os.path.exists(working_dir):
            print(f"⚠️  {domain_id} 的索引目录不存在，跳过（请先运行phase2_build_kg.py）")
            continue

        # 初始化RAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=custom_llm_complete,
            embedding_func=am_embedding
        )
        await rag.initialize_storages()

        # 按source两两组合查询
        source_pairs = list(combinations(valid_sources, 2))
        print(f"\n📋 共 {len(source_pairs)} 个source对需要查询")

        domain_matches = []
        for idx, (source_a, source_b) in enumerate(source_pairs):
            cols_a = source_columns[source_a]
            cols_b = source_columns[source_b]
            print(f"  [{idx+1}/{len(source_pairs)}] {source_a}({len(cols_a)}列) vs {source_b}({len(cols_b)}列)...")
            pairs = await query_source_pair(rag, source_a, source_b, cols_a, cols_b, domain_id)
            print(f"    → 发现 {len(pairs)} 个匹配对")
            domain_matches.extend(pairs)

            # 每10对保存一次中间结果
            if (idx + 1) % 10 == 0:
                all_matches.extend(domain_matches)
                domain_matches = []
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_matches, f, indent=2, ensure_ascii=False)
                print(f"    💾 中间结果已保存，当前共 {len(all_matches)} 个匹配对")

        all_matches.extend(domain_matches)
        await rag.finalize_storages()
        print(f"✅ {domain_id} 完成，本域发现 {len(domain_matches)} 个匹配对")

    # 最终保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_matches, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"🎯 全部完成！共发现 {len(all_matches)} 个匹配列对")
    print(f"📁 结果已保存至：{OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())