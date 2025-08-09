import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


def load_data(file: str, mode: str) -> List[Dict]:
    with open(file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if mode == "train":
        return [
            {
                "question": item["input"]["question"],
                "answer": item["output"]["answer"]
            } for item in raw
        ]
    elif mode == "test":
        return [
            {
                "question": item["input"]["question"]
            } for item in raw
        ]
    else:
        raise ValueError("mode must be 'train' or 'test'")


def tokenize_texts(texts: List[str], tokenizer) -> List[List[str]]:
    return [tokenizer.tokenize(text) for text in texts]


def embed_texts(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)


def retrieve_top_k_train_samples(
    query: str,
    query_embedding: np.ndarray,
    train_embeddings: np.ndarray,
    bm25: BM25Okapi,
    train_samples: List[Dict],
    top_k: int,
    alpha: float
) -> List[Dict]:
    dense_scores = cosine_similarity([query_embedding], train_embeddings)[0]
    bm25_scores = bm25.get_scores(query.split())

    scaler = MinMaxScaler()
    dense_scores = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()
    bm25_scores = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()

    combined_scores = alpha * dense_scores + (1 - alpha) * bm25_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [train_samples[i] for i in top_indices]

# test 셋 기반으로 train 데이터셋에서 퓨샷 추출
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_file', default='../data/korean_language_rag_V1.0_train.json')
    # parser.add_argument('--test_file', default='../data/korean_language_rag_V1.0_test.json')
    # parser.add_argument('--output_file', default='../data/few_shot_test_0.3.json')
    # # parser.add_argument('--embedding_model', default='kakaocorp/kanana-nano-2.1b-embedding')
    # parser.add_argument('--embedding_model', default='Qwen/Qwen3-Embedding-0.6b')
    # parser.add_argument('--top_k', type=int, default=3)
    # parser.add_argument('--alpha', type=float, default=0.3)
    
    # dev 셋 기반으로 train 데이터셋에서 퓨샷 추출
    parser.add_argument('--train_file', default='../data/korean_language_rag_V1.0_train.json')
    parser.add_argument('--test_file', default='../data/korean_language_rag_V1.0_dev.json')
    parser.add_argument('--output_file', default='../data/few_shot_dev_0.3.json')
    # parser.add_argument('--embedding_model', default='kakaocorp/kanana-nano-2.1b-embedding')
    parser.add_argument('--embedding_model', default='Qwen/Qwen3-Embedding-0.6b')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)

    args = parser.parse_args()

    print("Loading model")
    model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    print("Loading data")
    train_samples = load_data(args.train_file, mode="train")
    test_samples = load_data(args.test_file, mode="test")

    # train 데이터셋 question
    train_questions = [sample["question"] for sample in train_samples]
    tokenized_train_questions = tokenize_texts(train_questions, tokenizer)
    bm25 = BM25Okapi(tokenized_train_questions)
    train_embeddings = embed_texts(model, train_questions)

    print("Starting retrieval")
    results = []
    for i, sample in enumerate(test_samples, 1):
        test_q = sample["question"]
        test_embedding = embed_texts(model, [test_q])[0]

        top_samples = retrieve_top_k_train_samples(
            query=test_q,
            query_embedding=test_embedding,
            train_embeddings=train_embeddings,
            bm25=bm25,
            train_samples=train_samples,
            top_k=args.top_k,
            alpha=args.alpha
        )

        results.append({
            "question": test_q,
            "few_shots": top_samples
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Few-shot 예시 저장: {args.output_file}")


if __name__ == '__main__':
    main()
