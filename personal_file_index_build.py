# -*- coding: utf-8 -*-
"""
personal_file_index_build.py
从指定目录列表递归扫描 PDF / DOCX，抽取文本 -> 切分 -> Embedding -> 保存向量索引。
输出： ./rag_index/{faiss.index, corpus.json, metas.json}

用法：
    直接运行：python personal_file_index_build.py
    运行前请修改 root_list
"""

import os
import re
import json
import time
import glob
import math
import traceback
from typing import List, Tuple, Dict

# 文本与索引
import fitz  # PyMuPDF
from docx import Document
import numpy as np
import faiss
from tqdm import tqdm

# 向量模型
import torch
from sentence_transformers import SentenceTransformer

# ========================= 配置区 =========================
# 需要扫描的“根目录”列表——按你的实际电脑改这里
root_list: List[str] = [
    r"E:\嵌入式",
    r"C:\Users\tzh15\Downloads",
    r"C:\Users\tzh15\Desktop",
    r"C:\Users\tzh15\Documents"
    # r"D:\Projects\papers",
]

# 支持的文件后缀
EXTS = [".pdf", ".docx"]  # 如需支持 txt，可加入 ".txt"

# 最大文件大小（MB）——过大的 PDF/Word 多半是扫描件或含大量图片，先跳过以加速
MAX_FILE_MB = 200

# 切分参数（中文每字≈1 token/低估值，英文每词≈1.3~1.5 token 粗估）
CHUNK_LEN = 600      # 每个片段的最大字符数
CHUNK_OVERLAP = 120  # 片段重叠，保留上下文

# 嵌入模型（按需替换，更小更快可用 BAAI/bge-small-zh-v1.5）
EMB_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 批量大小（GPU 4060 可设 32~64；CPU 建议 8~16）
EMB_BATCH_SIZE = 64

# 输出目录
INDEX_DIR = "./rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# =========================================================


def readable_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"


def list_files(roots: List[str], exts: List[str]) -> List[str]:
    files = []
    for root in roots:
        # 递归匹配
        for ext in exts:
            pattern = os.path.join(root, "**", f"*{ext}")
            files.extend(glob.glob(pattern, recursive=True))
    # 去重、规范化
    normed = []
    seen = set()
    for p in files:
        q = os.path.normpath(p)
        if q.lower() not in seen:
            seen.add(q.lower())
            normed.append(q)
    return normed


def skip_too_large(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        return (size / (1024 * 1024)) > MAX_FILE_MB
    except Exception:
        return False


def extract_pdf(path: str) -> List[Tuple[int, str]]:
    """返回 [(page, text), ...]"""
    doc = fitz.open(path)
    out = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            out.append((i + 1, text))
    return out


def extract_docx(path: str) -> List[Tuple[int, str]]:
    """Word 视为单页返回 [(1, text)]"""
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


def build_corpus(files: List[str]) -> Tuple[List[str], List[Dict]]:
    corpus, metas = [], []
    print(f">>> 读取与切分阶段：共 {len(files)} 个文件")
    skipped_big = 0
    for path in tqdm(files, desc="读取文件"):
        try:
            if skip_too_large(path):
                skipped_big += 1
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = extract_pdf(path)
            elif ext == ".docx":
                pages = extract_docx(path)
            else:
                continue

            for page_no, content in pages:
                for ch in split_chunks(content):
                    corpus.append(ch)
                    metas.append({
                        "source": os.path.basename(path),
                        "path": os.path.abspath(path),
                        "page": int(page_no)
                    })
        except Exception as e:
            print(f"[跳过] {path}\n  原因: {e}")
            # 如需细节: print(traceback.format_exc())

    print(f">>> 切分完成：共 {len(corpus)} 个片段；跳过过大文件 {skipped_big} 个")
    return corpus, metas


def load_embed_model(name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 正在加载向量模型：{name}  设备：{device}")
    t0 = time.time()
    model = SentenceTransformer(name, device=device)
    print(f">>> 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return model


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = EMB_BATCH_SIZE) -> np.ndarray:
    print(f">>> 向量化阶段：共 {len(texts)} 个片段，batch_size={batch_size}")
    all_vecs = []
    t0 = time.time()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i: i + batch_size]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        all_vecs.append(vecs)
    arr = np.vstack(all_vecs).astype("float32")
    dt = time.time() - t0
    rate = len(texts) / dt if dt > 0 else float("inf")
    print(f">>> 向量化完成：{len(texts)} 条，用时 {dt:.1f}s，约 {rate:.1f} chunks/s")
    return arr


def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    dim = vecs.shape[1]
    print(f">>> 建立 FAISS 索引：向量数={vecs.shape[0]}, 维度={dim}")
    index = faiss.IndexFlatIP(dim)  # 余弦=内积（向量已归一化）
    t0 = time.time()
    index.add(vecs)
    print(f">>> 索引完成，用时 {time.time() - t0:.1f}s")
    return index


def save_outputs(index: faiss.Index, corpus: List[str], metas: List[Dict], out_dir: str = INDEX_DIR):
    faiss_path = os.path.join(out_dir, "faiss.index")
    corpus_path = os.path.join(out_dir, "corpus.json")
    metas_path = os.path.join(out_dir, "metas.json")

    print(">>> 保存索引与元数据 ...")
    faiss.write_index(index, faiss_path)
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    with open(metas_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    print(f"    - {faiss_path} ({readable_size(os.path.getsize(faiss_path))})")
    print(f"    - {corpus_path} ({readable_size(os.path.getsize(corpus_path))})")
    print(f"    - {metas_path}  ({readable_size(os.path.getsize(metas_path))})")
    print(">>> 全部完成！")


def main():
    print("=== 个人文件向量索引构建 开始 ===")
    print("根目录：")
    for r in root_list:
        print("  -", r)

    # 1) 扫描文件
    files = list_files(root_list, EXTS)
    print(f">>> 扫描到 {len(files)} 个候选文件")
    if not files:
        print("未发现可处理文件，请检查 root_list 或扩展名设置。")
        return

    # 2) 读取与切分
    corpus, metas = build_corpus(files)
    if not corpus:
        print("未抽取到文本内容；可能均为扫描件或受保护文件。")
        return

    # 3) 加载模型并向量化
    model = load_embed_model(EMB_MODEL_NAME)
    vecs = embed_texts(model, corpus, batch_size=EMB_BATCH_SIZE)

    # 4) 建索引
    index = build_faiss_index(vecs)

    # 5) 保存输出
    save_outputs(index, corpus, metas, out_dir=INDEX_DIR)

    print("=== 个人文件向量索引构建 结束 ===")


if __name__ == "__main__":
    main()
