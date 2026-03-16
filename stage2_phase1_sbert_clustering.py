import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_semantic_views(data_root):
    views = []
    for i in range(1, 19):
        sid = f"source{i}"
        path = os.path.join(data_root, sid, f"{sid}_semantic_view.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f); d['source_id'] = sid
                views.append(d)
    return views

def extract_text(v):
    desc = v.get('description', '')
    tables = [t['table_name'] for t in v.get('schema_inventory', [])]
    return f"{desc} Tables: {', '.join(tables)}"

def main():
    config = load_config()
    data_root = config['data_path']
    views = load_all_semantic_views(data_root)
    sids = [v['source_id'] for v in views]

    # 1. 向量化
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([extract_text(v) for v in views])

    # 2. 计算余弦距离矩阵
    sim_matrix = cosine_similarity(embeddings)
    dist_matrix = np.clip(1 - sim_matrix, 0, 2)

    # 3. 遍历距离阈值，用silhouette选最优
    best_threshold = None
    best_score = -1
    best_labels = None

    print("\n" + "="*50)
    print("🔍 遍历距离阈值，寻找最优聚类...")
    print("="*50)

    for threshold in np.arange(0.05, 0.8, 0.05):
        cluster_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='average'
        )
        lbls = cluster_model.fit_predict(dist_matrix)
        n = len(set(lbls))
        if n < 2 or n >= len(sids):
            print(f"threshold={threshold:.2f}  簇数={n}  (跳过)")
            continue
        cluster_sizes = [list(lbls).count(l) for l in set(lbls)]
        if min(cluster_sizes) < 2:
            print(f"threshold={threshold:.2f}  簇数={n}  最小簇大小={min(cluster_sizes)}  (有孤儿，跳过)")
            continue
        score = silhouette_score(dist_matrix, lbls, metric='precomputed')
        print(f"threshold={threshold:.2f}  簇数={n}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_labels = lbls

    if best_labels is None:
        print("\n⚠️  未找到满足条件的聚类结果，请调整阈值范围")
        return

    print(f"\n✅ 最优 threshold={best_threshold:.2f}，簇数={len(set(best_labels))}，silhouette={best_score:.4f}")
    labels = best_labels

    # 4. 整理结果
    unique_labels = sorted(set(labels))
    domains = {}
    for label in unique_labels:
        domains[f"domain_{label}"] = [sids[i] for i, l in enumerate(labels) if l == label]

    # 5. 输出结果
    print("\n" + "="*50)
    print(f"🎯 自动聚类结果：共发现 {len(unique_labels)} 个业务域")
    print("="*50)
    for d_id, members in sorted(domains.items()):
        print(f"{d_id}: {members}")

    output_path = os.path.join(data_root, "phase1_domains_clustering.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(domains, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 任务完成，结果已存至: {output_path}")

if __name__ == "__main__":
    main()