# -*- coding: utf-8 -*-
"""
find_path_llm.py
基于已构建索引（./rag_index/faiss.index, corpus.json, metas.json）做“按内容找文件”：
1) 本地向量检索（含路径/页码/片段/相似度）
2) 把证据喂给大模型，由模型输出：结论 + 候选文件路径列表 + 证据说明

用法：
  - 修改 main() 的 question
  - 配置你的 API_KEY / LLM_MODEL（硅基流动）
  - 运行：python find_path_llm.py
"""

import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI

# ========= 路径与模型配置 =========
INDEX_DIR = "./rag_index"
EMB_MODEL_NAME = "BAAI/bge-small-zh-v1.5"           # 必须与构建阶段一致；可换：BAAI/bge-small-zh-v1.5
TOPK = 60                                 # 初次召回数量
PER_FILE_LIMIT = 2                        # 每个文件保留的证据片段条数
MIN_SCORE = 0.10                          # 过滤阈值（0~1）
SNIPPET_LEN = 140                         # 片段预览长度（字符）

# ========= LLM（硅基流动 OpenAI 兼容）=========
API_KEY = "****************************************************"                  # ← 填你的硅基流动 API Key
BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"   # 或 DeepSeek-R1 / 其他你账号可用的模型
MAX_TOKENS = 700
TEMPERATURE = 0.2

# ========= 本地检索部分 =========
def load_index():
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    corpus_path = os.path.join(INDEX_DIR, "corpus.json")
    metas_path = os.path.join(INDEX_DIR, "metas.json")
    if not (os.path.exists(faiss_path) and os.path.exists(corpus_path) and os.path.exists(metas_path)):
        raise FileNotFoundError(
            f"缺少索引文件，请先构建：\n - {faiss_path}\n - {corpus_path}\n - {metas_path}"
        )
    index = faiss.read_index(faiss_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(metas_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    if len(corpus) != len(metas):
        raise ValueError(f"corpus 与 metas 数量不一致：{len(corpus)} vs {len(metas)}")
    return index, corpus, metas

def load_embed_model(name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 加载查询向量模型：{name}  设备：{device}")
    model = SentenceTransformer(name, device=device)
    return model

def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode([query], normalize_embeddings=True)
    return np.array(vec, dtype="float32")

def retrieve_raw(index, corpus, metas, qv: np.ndarray, topk: int):
    D, I = index.search(qv, topk)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "text": corpus[idx],
            "meta": metas[idx],
            "score": float(score)
        })
    return results

def aggregate_by_path(raw: List[Dict[str, Any]], per_file_limit: int, min_score: float) -> List[Dict[str, Any]]:
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw:
        path = r["meta"].get("path") or r["meta"].get("source")
        by_path.setdefault(path, []).append(r)

    ranked_paths = sorted(
        by_path.items(),
        key=lambda kv: max(x["score"] for x in kv[1]),
        reverse=True
    )

    merged: List[Dict[str, Any]] = []
    for path, items in ranked_paths:
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        # 过滤低分
        items = [x for x in items if x["score"] >= min_score]
        if not items:
            continue
        # 每文件最多保留 per_file_limit 条
        merged.extend(items[:per_file_limit])
    return merged

def shorten(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"

# ========= LLM 组装与调用 =========
def build_messages(question: str, hits: List[Dict[str, Any]]):
    """
    构造给 LLM 的 messages，包含明确的职责与输出格式。
    """
    system = (
        "你是一个本地文档检索助手。你的目标：\n"
        "1) 基于提供的【资料片段】判断哪些文件最可能匹配用户需求；\n"
        "2) 明确列出候选文件的【文件名】【完整路径】【页码】，并给出简短证据片段；\n"
        "3) 严禁编造路径或内容；如果证据不足，请直接回答“根据现有资料无法确定”。\n"
        "输出格式：\n"
        "- 先给 1-3 条要点结论；\n"
        "- 然后列出候选清单，每行包含：文件名 | 完整路径 | p.<页码> | 证据片段；\n"
        "- 若无结果，只输出那句固定话。"
    )

    # 把检索证据拼给模型（含路径）
    cites = []
    ctx_blocks = []
    for i, item in enumerate(hits, 1):
        meta = item["meta"]
        path = meta.get("path") or meta.get("source")
        page = meta.get("page", "-")
        src = f"{meta.get('source', os.path.basename(path))} | {path} | p.{page} | score={item['score']:.3f}"
        snippet = shorten(item["text"], 220)
        cites.append(f"[{i}] {src}")
        ctx_blocks.append(f"[{i}] {snippet}")

    user = (
        f"【问题】\n{question}\n\n"
        f"【命中片段】\n" + "\n".join(ctx_blocks) + "\n\n"
        f"【来源映射】\n" + "\n".join(cites) + "\n\n"
        "请按系统要求输出，不要加入未在来源中出现的路径。"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def call_llm(messages):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content

# ========= 主流程 =========
def main():
    # 直接在这里改问题（示例若与库匹配会返回路径）
    # question = "半年前我写的一份关于 XXX 的研究报告在哪？"
    # question = "我在研究迁移学习时写的PDF/Word文档存在哪里？"
    question = "帮我找一找电脑上我之前下载的关于使用机器学习方法对密集鱼苗计数的研究论文"

    # 1) 加载索引与模型
    index, corpus, metas = load_index()
    model = load_embed_model(EMB_MODEL_NAME)

    # 2) 本地语义检索
    qv = embed_query(model, question)
    raw = retrieve_raw(index, corpus, metas, qv, topk=TOPK)
    hits = aggregate_by_path(raw, per_file_limit=PER_FILE_LIMIT, min_score=MIN_SCORE)

    if not hits:
        print("根据现有索引没有足够匹配的结果。")
        print("建议：降低 MIN_SCORE、增大 TOPK，或检查构建是否包含该目录/文件类型。")
        return

    # 3) 构造 LLM 请求并调用
    messages = build_messages(question, hits)
    answer = call_llm(messages)

    # 4) 输出
    print("\n=== LLM 归纳结果 ===\n")
    print(answer)

if __name__ == "__main__":
    main()
