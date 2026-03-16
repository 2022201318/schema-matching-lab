# Schema Matching Lab

## 项目概述

模拟数据中心场景，对无标准目标表、包含多个domain的多个数据源的数据进行列级别的Schema Matching。

---

## 数据描述

| 数据源 | Benchmark | 数据属性 | Ground Truth |
|--------|-----------|---------|--------------|
| source1-2 | ChEMBL | 生物化学 | answer1 |
| source3-4 | OpenData | 公共政策/经济发展/财务审计 | answer2 |
| source5-6 | TPC-DI | 商业/消费者 | answer3 |
| source7-8 | TPC-DI | 人文艺术 | answer4 |
| source9-18 | GDC | 医学/临床数据 | answer5 |

**元数据说明：**
- source1-8：`s_metadata`（数据源级元数据）为人工标注，`t_metadata`（表级元数据）源自benchmark本身
- source9-18：`s_metadata`源于数据源引用的文章，`t_metadata`缺失

**Ground Truth说明：**
- answer1-4：两两匹配格式，已补全数据源信息
- answer5：以GDC目标表名为KEY，整合了10张源表的匹配关系，采用组展开方式评估

---

## 实验流程

### 阶段一：多层级元数据语义增强

对数据源级、表级、列级元数据分别进行语义增强，输出语义视图（semantic view）。

**数据源级：**
- 有描述则直接规则整理，不调用模型
- 无描述则调用Qwen-7B，输入表名和列名生成描述

**表级：**
- 调用Qwen-7B，输入列名、列样本、列属性生成表描述

**列级：**
- 调用Qwen-14B，输入列名、样本、属性及所属表和数据源生成语义描述
- 无法增强的输出`uncertain`，由72B模型兜底处理

所有语义视图存储在各数据源对应目录下。

---

### 阶段二：多层级候选筛选

**Step 1：Domain划分（数据源级）**

对数据源级语义视图做Embedding，使用层次聚类自动划分业务域，后续匹配只在domain内部进行，将O(n²)的复杂度压缩至domain内。
```
phase1_domains_clustering.json  →  domain_0 ~ domain_3
```

**Step 2：知识图谱构建（表级+列级）**

为每个domain单独构建LightRAG知识图谱，分三步注入：
- **骨架注入**：用规则直接注入 数据源→表→列 的层级归属关系，零LLM调用
- **语义文档注入**：将表级自然语言描述喂给LightRAG，由LLM抽取列间隐含语义关系
- **Embedding相似关系注入**：用SBERT计算列间余弦相似度，超过阈值的直接注入SIMILAR_TO关系边

**Step 3：候选列对生成**

对domain内每对source组合发起一次查询，LightRAG结合图结构和语义信息，批量输出两source间可能匹配的列对，输出为`final_matches.json`。

---

### 对比消融实验

| 实验 | 描述 | 输出文件 |
|------|------|---------|
| 完整方法 | 语义增强 + LightRAG图匹配 | `final_matches.json` |
| 无增强版本 | 仅列名 + LightRAG图匹配 | `final_matches_no_aug.json` |

---

## 实验结果

### 完整方法（有语义增强）

| 数据集 | Precision | Recall | F1 | Magneto-ft-llm Recall@GT |
|--------|-----------|--------|-----|--------------------------|
| ChEMBL | 0.8333 | 0.3125 | 0.4545 | ~0.790 |
| OpenData | 0.9643 | 0.7714 | 0.8571 | ~0.740 |
| TPC-DI | 0.9286 | 0.8667 | 0.8966 | ~0.740 |
| Wiki | 0.8571 | 1.0000 | 0.9231 | ~1.000 |
| GDC | 0.7263 | 0.5590 | 0.6317 | ~0.540 |
| ALL | 0.7310 | 0.5841 | 0.6494 | - |

### 无语义增强对比

| 数据集 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| ChEMBL | 0.8750 | 0.4375 | 0.5833 |
| OpenData | 0.8333 | 0.8571 | 0.8451 |
| TPC-DI | 1.0000 | 0.8667 | 0.9286 |
| Wiki | 0.8571 | 1.0000 | 0.9231 |
| GDC | 0.7000 | 0.5899 | 0.6402 |
| ALL | 0.7131 | 0.6215 | 0.6642 |

**结论：** 在列名本身语义丰富的数据集（ChEMBL、TPC-DI、GDC）上，语义增强边际收益为负，可能原因是LLM生成的通用描述引入了跨列的误匹配噪声；在列名风格多样的OpenData上，语义增强带来了精度提升。

---

## 主要代码文件

| 文件 | 说明 |
|------|------|
| `stage2_phase1_sbert_clustering.py` | 数据源级Embedding聚类，输出domain划分 |
| `phase2_build_kg.py` | 构建LightRAG知识图谱（有语义增强） |
| `phase2_build_kg_no_aug.py` | 构建LightRAG知识图谱（无语义增强） |
| `phase2_query_matches.py` | 查询生成候选匹配列对（有语义增强） |
| `phase2_query_matches_no_aug.py` | 查询生成候选匹配列对（无语义增强） |
| `evaluate.py` | 评估脚本，支持answer1-5两种GT格式，输出Precision/Recall/F1 |

---

## 毕设时间线

| 节点 | 时间 |
|------|------|
| 选题填报 | 2025年12月5日前 |
| 开题报告 | 2026年3月16日前 |
| 中期报告（论文初稿） | 2026年4月14日前 |
| 毕业论文终稿 | 2026年4月24日 |
| 第一次答辩 | 2026年5月初 |
