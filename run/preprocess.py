import json
import re

def clean_text(text):
    junk_token_pattern = re.compile(r'<\|.*?\|>')
    clean = re.sub(junk_token_pattern, '', text)
    to_remove = [
        "assistant",
        "한국어 문법 전문가입니다. 주어진 규칙을 참고하여 문제에 답변해 드리겠습니다.",
        "[|endofturn|]", "<|end_of_text|>", "<|begin_of_text|>",
        "<|im_end|>", "<|im_start|>", "<|start_header_id|>",
        "<|end_header_id|>", "<|start_of_text|>", "<|end_of_text|>",
        "<|eot_id|>","<s>","</s>","User: ","[수정된 문장]","[이유 설명] "
    ]
    for t in to_remove:
        clean = clean.replace(t, '')
    
    clean = clean.replace("**정답:","### 최종 답변:")
    clean = clean.replace("### 교정 결과","### 최종 답변:")
    clean = clean.replace("### 최종답변","### 최종 답변:")

    print(f"전처리 전: {clean}")

    if "### 최종 답변" in clean:
        clean = clean.split("### 최종 답변", maxsplit=1)[1]
    else:
        # "### 최종 답변"이 없는 경우, 아무 처리도 하지 않거나 빈 문자열을 반환할 수 있습니다.
        # 여기서는 원본 코드를 존중하여 그대로 두거나 혹은 빈 값으로 처리할 수 있습니다.
        # 안전하게 빈 문자열로 처리하려면 아래 주석을 해제하세요.
        # clean = ""
        pass

    clean = clean.replace(":", "") # 콜론 제거
    
    print(f"전처리 후: {clean}", f"길이: {len(clean)}") # 디버깅용 print
    print("-"*50)

    return clean.strip()

def save_as_json(data, filename):
    """
    데이터(리스트)를 표준 JSON 파일로 저장합니다.
    json.dump()는 전체 파이썬 객체를 파일에 직접 씁니다.
    - ensure_ascii=False: 한글이 깨지지 않고 그대로 저장되도록 합니다.
    - indent=4: 사람이 파일을 열어봤을 때 읽기 쉽도록 4칸 들여쓰기를 적용합니다.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_and_save_json(input_json_path, output_json_path):
    raw_data = load_json(input_json_path)
    
    for item in raw_data:
        answer = item.get("output", {}).get("answer", "")
        item["output"]["answer"] = clean_text(answer)

    save_as_json(raw_data, output_json_path)
    print(f"✅ 리스트 형태의 JSON 저장 완료: {output_json_path}")

process_and_save_json(
    '../result/믿습니다_3.json',
    '../result/믿습니다_3_clean.json'
)