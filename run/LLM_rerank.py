import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import json
from tqdm import tqdm
import gc

# 모델 이름과 경로 설정
def init_model():
    """
    Hugging Face Hub에서 모델을 로드합니다.
    CPU 오프로딩 없이, 모델 전체를 단일 GPU에 올리도록 강제합니다.
    """
    model_name = "K-intelligence/Midm-2.0-Base-Instruct"

    # 4비트 양자화
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quantization_config,
        device_map="cuda:0", # "auto" 대신 특정 GPU를 명시합니다.
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_name)

    print("모델 로딩 및 파이프라인 설정 완료! (단일 GPU)")
    return model,tokenizer,generation_config

# --- 1단계: 전체 데이터셋 로드 (이하 코드는 이전과 동일) ---
def load_full_dataset(file_path: str) -> list:
    print(f"\n📂 데이터셋 파일을 로드합니다: {file_path}")
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            dataset = json.load(f)
        print(f" 총 {len(dataset)}개의 항목을 성공적으로 불러왔습니다.")
        return dataset
    except Exception as e:
        print(f" 오류: 데이터셋 로딩 중 문제가 발생했습니다 - {e}")
        return []

# --- 2단계: 최적의 문서 집합 선택 ---
def select_best_document_indices(query: str, candidate_docs: list[str], model,tokenizer,generation_config) -> list[int]:
    formatted_documents = "\n".join([f"[문서 {i}]: {doc}" for i, doc in enumerate(candidate_docs)])
    prompt = f"""당신은 주어진 [질문]에 답하는 데 가장 관련성이 높은 문서들을 [후보 문서] 중에서 선택하는 한국어 언어 규범 전문가입니다.

당신의 임무는 다음과 같은 사고 과정에 따라, 최종적으로 선택된 문서들의 번호를 지정된 형식으로 출력하는 것입니다. 

1. **핵심 원리 파악**: [질문]을 해결하는 데 필요한 문법적 원리가 무엇인지 파악합니다. 만약 외국어에 대한 표기법일 경우에는 외래어 표기법에 관련된 조항을 살펴서 파악합니다.
2. **문서 매핑**: 파악된 원리를 가장 잘 설명하는 문서를 [후보 문서]에서 모두 찾습니다. 문서에 대한 설명의 경우 이 부분에서 설명하도록 합니다.
3. **최종 선택**: 찾은 문서들의 번호를 `### 최종 선택:` 형식으로 오롯이 문서 번호만을 한 줄에 출력합니다. 후보문서 중에서 정답문서를 반드시 한 개 이상 세 개 이하로 선택해서 출력해야 합니다. 최종 선택에서 아무것도 선택하지 않는 것은 안됩니다.
"""

    user_input = f"**[질문]** {query} **[후보 문서]**{formatted_documents} "
    message = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ]
    source = tokenizer.apply_chat_template(
                    message,
                    tokenize = True,
                    force_reasoning=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
    outputs = model.generate(
                            source.to("cuda:0"),
                            generation_config=generation_config,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=1024,
                            do_sample=False,
                        )
    generated_texts = []
    for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=False)
            generated_texts.append(text)
    assistant_response = generated_texts[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
    print(assistant_response)
    selection_line = assistant_response.split('### 최종 선택:')[1].strip()
    selection_line = selection_line.split("**설명")[0]
    selection_line = selection_line.split("**수정")[0]
    selection_line = selection_line.split("\n")[0]
    selection_line = selection_line.replace("문서","").strip()
    selection_line = selection_line.replace(" ###","").strip()
    selection_line = selection_line.replace("```","").strip()
    selection_line = selection_line.replace("[","").strip()
    selection_line = selection_line.replace("]","").strip()
    selection_line = selection_line.replace("*","").strip()
    selection_line = selection_line.replace("<|end_of_text|>","").strip()
    selection_line = selection_line.replace(", "," ").strip()
    selection_line = selection_line.replace("\n"," ").strip()
    selection_doc = selection_line.split(" ")
    print(selection_doc)
    print("seletion_line",len(selection_doc))
    

    selected_indices_str = selection_doc
    return [int(i) for i in selected_indices_str if i != ""]
    # return [i for i in selected_indices_str]


# --- 결과 저장 함수 ---
def save_results_to_json(results: list, output_file_path: str):
    print(f"\n 총 {len(results)}개의 처리 결과를 파일에 저장합니다: {output_file_path}")
    try:
        with open(output_file_path, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(" 파일 저장이 완료되었습니다.")
    except Exception as e:
        print(f"오류: 결과를 파일에 저장하는 중 문제가 발생했습니다 - {e}")

# --- 메인 실행 로직 (배치 크기 1 보장을 위한 메모리 정리 포함) ---
def main():
    model,tokenizer,generation_config = init_model()
    
    # input_file_path = "/home/nlplab/ssd2/jeong/RAG/GCU-koreanRAG/data/믿습니다_3_chunks_test_0.6_qwen8b_k25.json"
    input_file_path = "../result/믿오없5_쿼리분할20_top25.json"

    dataset = load_full_dataset(input_file_path)

    if not dataset:
        print("데이터셋이 비어 있어 프로그램을 종료합니다.")
        return

    all_results = []
    
    print("\n" + "="*80)
    print("🚀 전체 데이터셋에 대한 Set Selection(Reranking)을 시작합니다. (배치 크기 1)")
    print("="*80)
    
    for item in tqdm(dataset, desc="Processing Queries"):
        # query = item.get("question") or item.get("query")
        query = item.get("answer") or item.get("query")
        original_chunks = item.get("retrieved_chunks", [])
        
        if not query or not original_chunks:
            continue

        candidate_docs_text = [chunk['text'] for chunk in original_chunks]
        selected_indices = select_best_document_indices(
            query, candidate_docs_text, model,tokenizer,generation_config
        )
        selected_texts = [candidate_docs_text[i] for i in selected_indices if i < 25]
        result_item = {
            "question": query,
            "answer": item.get("answer"),
            "reranking_result": {
                "selected_indices": selected_indices,
                "selected_texts": selected_texts
            },
            "original_retrieved_chunks": original_chunks
        }
        all_results.append(result_item)

            # gc.collect()
            # torch.cuda.empty_cache()

    output_file_path = "../result/reranking_results_test.json"
    save_results_to_json(all_results, output_file_path)

if __name__ == "__main__":
    main()