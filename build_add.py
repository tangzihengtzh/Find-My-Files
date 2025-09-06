# -*- coding: utf-8 -*-
"""
build_add.py
在已有向量库基础上，增量扫描 root_list 下“新增”的 PDF/DOCX 文件，并把新向量追加到索引。
要求已有 ./rag_index/ 下的三件套：faiss.index / corpus.json / metas.json
首次也可用：若三件套不存在，则会自动创建为“空库”再增量导入。

局限：若已入库文件内容发生变化，本脚本不会更新/删除旧向量；需要全量重建或扩展“脏文件重建”逻辑。
"""

import os
import re
import json
import time
import glob
import traceback
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
from docx import Document
import numpy as np
import faiss
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

# ========================= 配置区 =========================
root_list: List[str] = [
    r"C:\Users\你的用户名\Desktop",
    r"E:\研一文献阅读",
    # r"D:\Projects\papers",
]
EXTS = [".pdf", ".docx"]

MAX_FILE_MB = 200
CHUNK_LEN = 600
CHUNK_OVERLAP = 120

EMB_MODEL_NAME = "BAAI/bge-m3"          # 与已有库保持一致（非常重要）
EMB_BATCH_SIZE = 64

INDEX_DIR = "./rag_index"
STATE_PATH = os.path.join(INDEX_DIR, "index_state.json")  # 记录已入库文件的 path/size/mtime/chunk_count
os.makedirs(INDEX_DIR, exist_ok=True)
# =========================================================


def list_files(roots: List[str], exts: List[str]) -> List[str]:
    files = []
    for root in roots:
        for ext in exts:
            pattern = os.path.join(root, "**", f"*{ext}")
            files.extend(glob.glob(pattern, recursive=True))
    # 去重规范化
    seen = set()
    out = []
    for p in files:
        q = os.path.normpath(p)
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def readable_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"


def skip_too_large(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        return (size / (1024 * 1024)) > MAX_FILE_MB
    except Exception:
        return False


def extract_pdf(path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(path)
    out = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            out.append((i + 1, text))
    return out


def extract_docx(path: str) -> List[Tuple[int, str]]:
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return [(1, text)] if text else []


def split_chunks(text: str, max_len: int = CHUNK_LEN, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks, start = [], 0
    N = len(text)
    while start < N:
        end = min(N, start + max_len)
        chunks.append(text[start:end])
        if end == N:
            break
        start = max(0, end - overlap)
    return chunks


def load_or_init_store():
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    corpus_path = os.path.join(INDEX_DIR, "corpus.json")
    metas_path = os.path.join(INDEX_DIR, "metas.json")

    # 载入/初始化索引
    if os.path.exists(faiss_path) and os.path.exists(corpus_path) and os.path.exists(metas_path):
        index = faiss.read_index(faiss_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        with open(metas_path, "r", encoding="utf-8") as f:
            metas = json.load(f)
        print(f">>> 载入现有索引：{len(corpus)} 个片段")
    else:
        # 空库
        print(">>> 未检测到现有索引，将初始化空库并进行增量导入")
        # 先创建一个 0 向量的 Index；维度要在首次嵌入后重建，这里先占位
        index = None
        corpus, metas = [], []
    # 载入/初始化状态
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"files": {}}  # path -> {size, mtime, chunk_count}
    return index, corpus, metas, state


def save_store(index, corpus, metas, state):
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    corpus_path = os.path.join(INDEX_DIR, "corpus.json")
    metas_path = os.path.join(INDEX_DIR, "metas.json")

    # 保存索引与数据
    if index is not None:
        faiss.write_index(index, faiss_path)
        print(f"    - 保存 {faiss_path} ({readable_size(os.path.getsize(faiss_path))})")

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"    - 保存 {corpus_path} ({readable_size(os.path.getsize(corpus_path))})")

    with open(metas_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    print(f"    - 保存 {metas_path} ({readable_size(os.path.getsize(metas_path))})")

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    print(f"    - 更新 {STATE_PATH}")


def load_embed_model(name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 加载向量模型：{name}  设备：{device}")
    t0 = time.time()
    model = SentenceTransformer(name, device=device)
    print(f">>> 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return model


def embed_texts(model: SentenceTransformer, texts, batch_size: int) -> np.ndarray:
    print(f">>> 向量化新增片段：{len(texts)}，batch_size={batch_size}")
    all_vecs = []
    t0 = time.time()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding(new)"):
        batch = texts[i: i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        all_vecs.append(vecs)
    arr = np.vstack(all_vecs).astype("float32") if all_vecs else np.zeros((0, 1), dtype="float32")
    dt = time.time() - t0
    if len(texts) > 0:
        print(f">>> 向量化完成：{len(texts)} 条，用时 {dt:.1f}s，约 {len(texts)/max(dt,1e-6):.1f} chunks/s")
    return arr


def ensure_index(dim: int, index):
    """若现有 index 为空（首次导入），创建新 Index；否则返回原 index。"""
    if index is not None:
        return index
    print(f">>> 初始化 FAISS 索引（维度={dim}）")
    return faiss.IndexFlatIP(dim)


def main():
    print("=== 增量构建（新增文件）开始 ===")
    for r in root_list:
        print(" 根目录：", r)

    index, corpus, metas, state = load_or_init_store()

    # 1) 扫描候选
    files = list_files(root_list, EXTS)
    print(f">>> 扫描到 {len(files)} 个候选文件")

    # 2) 过滤（仅新增）
    new_files = []
    for path in files:
        try:
            if skip_too_large(path):
                continue
            st = os.stat(path)
            size = st.st_size
            mtime = int(st.st_mtime)
            rec = state["files"].get(os.path.abspath(path))
            if rec is None:
                new_files.append((path, size, mtime))
        except Exception:
            continue

    print(f">>> 新增文件数：{len(new_files)}")
    if not new_files:
        print(">>> 没有新增文件，无需更新索引。")
        return

    # 3) 读取+切分新增
    new_corpus, new_metas = [], []
    skip_err = 0
    for path, size, mtime in tqdm(new_files, desc="读取新增文件"):
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = extract_pdf(path)
            elif ext == ".docx":
                pages = extract_docx(path)
            else:
                continue
            chunk_count = 0
            for page_no, content in pages:
                for ch in split_chunks(content):
                    new_corpus.append(ch)
                    new_metas.append({
                        "source": os.path.basename(path),
                        "path": os.path.abspath(path),
                        "page": int(page_no)
                    })
                    chunk_count += 1
            # 暂不写入 state，等成功嵌入+入库后再更新
            # 临时把 chunk_count 记在 tuple 的第 4 位
            # 我们在入库成功后写回真实 chunk_count
        except Exception as e:
            skip_err += 1
            print(f"[跳过] {path}\n  原因: {e}")
            # print(traceback.format_exc())

    print(f">>> 新增切分片段：{len(new_corpus)} 条；解析失败：{skip_err} 个文件")
    if len(new_corpus) == 0:
        print(">>> 新增文件没有可抽取文本，结束。")
        return

    # 4) 嵌入新增
    model = load_embed_model(EMB_MODEL_NAME)
    new_vecs = embed_texts(model, new_corpus, EMB_BATCH_SIZE)
    if new_vecs.shape[0] == 0:
        print(">>> 无新增向量，结束。")
        return

    # 5) 将新增向量追加到索引
    index = ensure_index(new_vecs.shape[1], index)
    before = index.ntotal if hasattr(index, "ntotal") else 0
    index.add(new_vecs)
    after = index.ntotal if hasattr(index, "ntotal") else before + new_vecs.shape[0]
    print(f">>> 追加向量完成：{before} -> {after} 条")

    # 6) 追加文本与元数据，并更新 state
    start_len = len(corpus)
    corpus.extend(new_corpus)
    metas.extend(new_metas)

    # 为每个新增文件统计 chunk 数（从 new_metas 聚合）
    chunks_per_path: Dict[str, int] = {}
    for m in new_metas:
        p = m["path"]
        chunks_per_path[p] = chunks_per_path.get(p, 0) + 1

    for path, size, mtime in new_files:
        pabs = os.path.abspath(path)
        cnt = chunks_per_path.get(pabs, 0)
        if cnt > 0:
            state["files"][pabs] = {
                "size": size,
                "mtime": mtime,
                "chunk_count": cnt
            }

    # 7) 保存
    print(">>> 保存更新：")
    save_store(index, corpus, metas, state)

    print("=== 增量构建完成 ===")


if __name__ == "__main__":
    main()
