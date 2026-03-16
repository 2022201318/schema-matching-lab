import json
import os
from itertools import combinations

# ==================== 配置 ====================
PRED_FILE = "data/final_matches_no_aug.json"
GT_DIR = "ground_truth"

# ==================== 加载预测 ====================
def load_pred_pairs():
    with open(PRED_FILE, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    pred_pairs = set()
    for item in pred_data:
        pair = tuple(sorted([item['col_a'], item['col_b']]))
        pred_pairs.add(pair)
    return pred_pairs

# ==================== 加载单个GT文件 ====================
def load_single_gt(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_pairs = set()

    # answer1-4格式
    if isinstance(data, dict) and 'matches' in data:
        for m in data['matches']:
            col_a = f"{m['source_id']}.{m['source_table']}.{m['source_column']}"
            col_b = f"{m['target_id']}.{m['target_table']}.{m['target_column']}"
            gt_pairs.add(tuple(sorted([col_a, col_b])))

    # answer5格式
    elif isinstance(data, list):
        for anchor in data:
            cols = [
                f"{c['source_id']}.{c['table']}.{c['column']}"
                for c in anchor['aligned_columns']
            ]
            for col_a, col_b in combinations(cols, 2):
                if col_a.split('.')[0] != col_b.split('.')[0]:
                    gt_pairs.add(tuple(sorted([col_a, col_b])))

    return gt_pairs

# ==================== 评估 ====================
def evaluate(gt_pairs, pred_pairs, label):
    # 只看和这个GT相关的预测（pred里的列都在GT涉及的source范围内）
    gt_sources = set()
    for a, b in gt_pairs:
        gt_sources.add(a.split('.')[0])
        gt_sources.add(b.split('.')[0])

    # 只保留两列都在GT source范围内的预测对
    relevant_pred = {
        (a, b) for a, b in pred_pairs
        if a.split('.')[0] in gt_sources and b.split('.')[0] in gt_sources
    }

    tp = gt_pairs & relevant_pred
    fp = relevant_pred - gt_pairs
    fn = gt_pairs - relevant_pred

    precision = len(tp) / len(relevant_pred) if relevant_pred else 0
    recall    = len(tp) / len(gt_pairs)      if gt_pairs      else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0)

    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"GT涉及的source : {sorted(gt_sources)}")
    print(f"GT列对总数     : {len(gt_pairs)}")
    print(f"相关预测总数   : {len(relevant_pred)}  (从{len(pred_pairs)}条中过滤)")
    print(f"TP (命中)      : {len(tp)}")
    print(f"FP (误报)      : {len(fp)}")
    print(f"FN (漏报)      : {len(fn)}")
    print(f"{'='*60}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1             : {f1:.4f}")

    if fn:
        print(f"\n🔍 漏报样例（前5条）：")
        for pair in list(fn)[:5]:
            print(f"  {pair[0]}  <->  {pair[1]}")
    if fp:
        print(f"\n🔍 误报样例（前5条）：")
        for pair in list(fp)[:5]:
            print(f"  {pair[0]}  <->  {pair[1]}")

    return precision, recall, f1

# ==================== 主流程 ====================
def main():
    print("📂 加载预测结果...")
    pred_pairs = load_pred_pairs()
    print(f"✅ 预测共 {len(pred_pairs)} 条列对")

    all_gt_pairs = set()
    results = {}

    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(GT_DIR, fname)
        gt_pairs = load_single_gt(fpath)
        all_gt_pairs |= gt_pairs
        label = fname.replace('.json', '')
        p, r, f1 = evaluate(gt_pairs, pred_pairs, label)
        results[label] = (p, r, f1)

    # 汇总
    print(f"\n{'='*60}")
    print(f"📊 汇总（全部GT合并）")
    print(f"{'='*60}")
    p, r, f1 = evaluate(all_gt_pairs, pred_pairs, "ALL")
    results['ALL'] = (p, r, f1)

    # 表格汇总
    print(f"\n{'='*60}")
    print(f"{'文件':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*60}")
    for name, (p, r, f) in results.items():
        print(f"{name:<12} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()