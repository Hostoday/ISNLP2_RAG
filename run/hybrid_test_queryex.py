# Example이 여러 개일 때 각각 text + 예시로 분리해서 검색하도록 수정한 버전
# 쿼리 재작성 추가 -> 총 25개

import os
import re
import gc
import json
import argparse
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import BitsAndBytesConfig


os.environ.setdefault("HF_HUB_CACHE", "/home/nlplab/ssd1/hyunmin/code/model_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")


try:
    from konlpy.tag import Mecab  # type: ignore
    _mecab = Mecab()

    def ko_morphs(text: str) -> List[str]:
        return _mecab.morphs(text)
except Exception:  # Mecab not available
    _re_tok = re.compile(r"[가-힣]+|[A-Za-z0-9]+")

    def ko_morphs(text: str) -> List[str]:  # fallback
        return _re_tok.findall(text)


def parse_examples(examples: Union[str, List[str]]) -> List[str]:
    if not examples:
        return []
    examples = [examples] if isinstance(examples, str) else examples
    out: List[str] = []
    for ex in examples:
        ex = ex.strip()
        if not ex:
            continue
        if len(ex) > 200:
            out.extend([e.strip() for e in ex.split(',') if e.strip()])
        else:
            out.append(ex)
    return out

_reason_pat = re.compile(r"(?:이|가) 옳다[.\"\']?\s*(.*)")

def extract_reason_text(answer: str) -> str:
    m = _reason_pat.search(answer)
    return (m.group(1) if m else answer).strip()


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 16) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)


def load_chunks_subdocs(chunk_file: str) -> Tuple[List[Dict], List[str], List[List[str]], List[int]]:
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)

    sub_docs, sub_tokens, sub2cid = [], [], []
    for cid, c in enumerate(chunks):
        base = f"{c['rule_title']}: {c['text']}".strip()
        examples = parse_examples(c.get("examples", [])) or [""]
        for ex in examples:
            doc = base if not ex else f"{base} ▶ {ex}"
            sub_docs.append(doc)
            sub_tokens.append(ko_morphs(doc))
            sub2cid.append(cid)
    return chunks, sub_docs, sub_tokens, sub2cid


def load_train_data(train_file: str) -> List[Dict]:
    with open(train_file, encoding="utf-8") as f:
        data = json.load(f)
    return [{"question": d["input"]["question"], "answer": d["output"]["answer"]} for d in data]


def retrieve_top_k_chunks_maxpool(
    query: str,
    answer_emb: np.ndarray,
    sub_embs: np.ndarray,
    bm25: BM25Okapi,
    sub2cid: List[int],
    chunks: List[Dict],
    k: int = 3,
    alpha: float = 0.7,
) -> List[int]:
    dense_scores = cosine_similarity([answer_emb], sub_embs)[0]
    bm25_scores = bm25.get_scores(ko_morphs(query))

    # Min‑Max normalisation (NumPy ≥2.0 safe)
    d_min, d_max = dense_scores.min(), dense_scores.max()
    b_min, b_max = min(bm25_scores), max(bm25_scores)

    dense_n = (dense_scores - d_min) / (np.ptp(dense_scores) + 1e-9)
    bm25_n = (bm25_scores - b_min) / (b_max - b_min + 1e-9)

    hybrid = alpha * dense_n + (1 - alpha) * bm25_n

    best: Dict[int, float] = {}
    for sid, score in enumerate(hybrid):
        cid = sub2cid[sid]
        if score > best.get(cid, -1):
            best[cid] = score

    return sorted(best, key=best.get, reverse=True)[:k]


def find_best_examples(
    answer: str,
    examples: List[str],
    model: SentenceTransformer,
    top_k: int = 3,
    alpha: float = 0.2,
    batch_size: int = 16,
) -> List[str]:
    if not examples:
        return []
    if len(examples) > 50:
        answer_tokens = ko_morphs(answer)
        bm25 = BM25Okapi([ko_morphs(e) for e in examples])
        idx = np.argsort(bm25.get_scores(answer_tokens))[::-1][:20]
        examples = [examples[i] for i in idx]

    ans_emb = embed_texts(model, [answer], 1)[0]
    ex_embs = embed_texts(model, examples, batch_size)
    dense = cosine_similarity([ans_emb], ex_embs)[0]

    bm25_scores = BM25Okapi([ko_morphs(e) for e in examples]).get_scores(ko_morphs(answer))

    d_n = (dense - dense.min()) / (np.ptp(dense) + 1e-9)
    b_n = (bm25_scores - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-9)
    final = alpha * d_n + (1 - alpha) * b_n

    return [examples[i] for i in np.argsort(final)[::-1][:min(top_k, len(examples))]]


def main():
    ap = argparse.ArgumentParser()
    
    # 테스트 파일
    ap.add_argument("--test_file",   default="../result/믿습니다_3_clean.json")
    # 쿼리 재작성 파일
    ap.add_argument("--query_file",  default="../result/믿음아오류없이해라.json")
    ap.add_argument("--top_k_test",  type=int, default=20)
    ap.add_argument("--top_k_query", type=int, default=5)
    
    ap.add_argument("--chunk_file",  default="../data/korean_rule_chunks_final4.json")
    # output
    ap.add_argument("--output_file", default="../result/믿오없5_쿼리분할20_top25.json")
    

    ap.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--disable_dynamic_examples", action="store_true")
    ap.add_argument("--max_examples_per_chunk", type=int, default=100)
    ap.add_argument("--clear_cache_interval", type=int, default=50)
    args = ap.parse_args()

    # ────────── (2) 모델 준비 동일) ──────────
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = SentenceTransformer(args.embedding_model, trust_remote_code=True,
                                model_kwargs={"quantization_config": qcfg, "device_map": "cuda:0"})

    # ────────── (3) 청크 서브‑도큐먼트 로드 및 인덱싱) ──────────
    print("[1/6] Loading chunks …")
    chunks, sub_docs, sub_tokens, sub2cid = load_chunks_subdocs(args.chunk_file)

    print("[2/6] Encoding sub‑docs …")
    sub_embs = embed_texts(model, sub_docs)

    print("[3/6] Building BM25 index …")
    bm25 = BM25Okapi(sub_tokens)

    # ────────── (4) 테스트/재작성 쿼리 파일 로드) ──────────
    tests   = load_train_data(args.test_file)          # [{'question', 'answer'} …]
    queries = load_train_data(args.query_file)
    assert len(tests) == len(queries), "두 파일 샘플 수 다름"

    results = []
    for i, (t_samp, q_samp) in enumerate(zip(tests, queries)):
        if i % 20 == 0:
            print(f"  processing {i+1}/{len(tests)} …")
        if i and i % args.clear_cache_interval == 0:
            torch.cuda.empty_cache(); gc.collect()

        # ── (4‑1) test_file ‑‑> 상위 20 ───────────────────
        reason_t   = extract_reason_text(t_samp["answer"])
        ans_emb_t  = embed_texts(model, [reason_t], 1)[0]
        cids_test  = retrieve_top_k_chunks_maxpool(
                        t_samp["answer"], ans_emb_t, sub_embs, bm25,
                        sub2cid, chunks, k=args.top_k_test, alpha=args.alpha)

        # ── (4‑2) query_file ‑‑> 상위 5 ───────────────────
        reason_q   = extract_reason_text(q_samp["answer"])
        ans_emb_q  = embed_texts(model, [reason_q], 1)[0]
        cids_query = retrieve_top_k_chunks_maxpool(
                        q_samp["answer"], ans_emb_q, sub_embs, bm25,
                        sub2cid, chunks, k=args.top_k_query, alpha=args.alpha)

        # ── (4‑3) 중복 제거 + 최대 25개 유지 ───────────────
        merged_cids = []
        for cid in cids_test + cids_query:
            if cid not in merged_cids:
                merged_cids.append(cid)
            if len(merged_cids) == 25:
                break

        # ── (4‑4) 청크 상세 + 예시 선별 ───────────────────
        top_chunks = []
        for cid in merged_cids:
            c = chunks[cid].copy()
            if not args.disable_dynamic_examples:
                ex_parsed = parse_examples(c.get("examples", []))[: args.max_examples_per_chunk]
                best_ex   = find_best_examples(reason_t, ex_parsed, model, 3) or ["예시 없음"]
                c["examples"] = best_ex
            top_chunks.append({"text": c["text"], "examples": c["examples"]})

        results.append({
            "question": t_samp["question"],
            "answer":   t_samp["answer"],
            "retrieved_chunks": top_chunks
        })

    # ────────── (5) 저장 ──────────
    with open(args.output_file, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)
    print("[6/6] Retrieval done →", args.output_file)

if __name__ == "__main__":
    main()