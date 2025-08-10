import json

split = "test"

with open(f'korean_language_rag_V1.0_{split}.json', 'r') as f:
    data = json.load(f)


with open(f'reranking_results_{split}.json', 'r') as f:
    reranking_data = json.load(f)

for d, rerank in zip(data, reranking_data):
    selected_indices = rerank['reranking_result']['selected_indices'][::-1]
    retrieval = []
    for idx in selected_indices:
        r = rerank["original_retrieved_chunks"][idx]
        text = r["text"]
        example = "\n".join(r["examples"])
        retrieval.append(text + "\n" + example)
    d.update({'retrieval': retrieval})

with open(f'../data/{split}_with_examples.json', 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)