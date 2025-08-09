#!/bin/sh

if [ -f "environment.yml" ]; then
  echo "Creating Conda environment..."

  conda env create -f environment.yml
  conda env create -f environment_rag.yml

conda activate rag_min

python -m run.chunking

python -m run.fewshotfinder


python -m run.test\
    --output "믿습니다_3.json"\
    --model_id "K-intelligence/Midm-2.0-Base-Instruct"\
    --model_ckpt_path ""\
    --device cuda:0\
    --batch_size 2\
    --prompt_type "few_example_instruct_new"\
    --prompt "당신은 국립국어원 최고의 언어학자입니다. 다음 문장을 분석하여 어문 규범에 맞게 수정하고, 어떤 문법적 원칙에 따라 수정했는지 그 이유를 상세히 설명해주십시오.다음 규칙을 반드시 준수하십시오: 1. \"### 최종답변:\"을 작성하고 그 뒤에 교정된 문장과 설명을 출력해야 합니다. 2. 예시와 같은 출력의 문장 구조를 띄어야 합니다."\
    --unsloth None

python -m run.preprocess

read -r -d '' SYSTEM_PROMPT <<'EOF'
당신은 국어 규범 진단 및 검색 전략가(Grammar Diagnostician and Search Strategist)입니다. 당신의 임무는 주어진 [문제]의 문장을 분석하여 문법적 오류를 진단하고, 그 진단을 바탕으로 해결에 필요한 규정 정보를 찾기 위한 최적의 검색어들을 설계하는 것입니다.

**[핵심 업무 원칙]**
1.  **선 진단, 후 설계:** 반드시 문장의 오류를 먼저 분석하고, 그 분석 내용에 근거하여 검색어를 설계해야 합니다.
2.  **다각적 검색:** 생성된 검색어들은 각각 '조항 제목', '핵심 원리', '구체적 용례'를 찾아낼 수 있도록 다각적으로 구성되어야 합니다.
3.  **출력 형식 엄수:** 당신의 **최종적이고 유일한 출력물**은 지정된 JSON 형식이어야 합니다. 분석 과정 등 다른 어떤 텍스트도 절대 포함해서는 안 됩니다.

**[내부 사고 과정 가이드라인]**
당신은 검색어를 출력하기 전, 머릿속으로 다음의 사고 과정을 거쳐야 합니다.

*   **1단계: 문제 진단 (Problem Diagnosis)**
    *   **오류 의심 부분:** [문제]의 문장에서 문법적으로 어색하거나 틀린 부분은 어디인가?
    *   **쟁점 분류:** 이 오류는 어떤 문법 범주에 속하는가? (예: 띄어쓰기(의존 명사), 띄어쓰기(보조 용언), 표준어 규정 등)
    *   **분석 내용:** 왜 이 부분이 틀렸다고 생각하는가? 어떤 규칙이 적용되어야 할 것 같은가?

*   **2단계: 검색 전략 수립 (Search Strategy Formulation)**
    *   **조항 제목 검색:** 진단 결과를 바탕으로, 이 문제를 해결할 수 있는 가장 정확한 `rule_title`은 무엇일까?
    *   **규정 내용 검색:** 이 문제의 핵심 원리를 설명하는 `text`를 찾으려면 어떤 키워드로 검색해야 할까?
    *   **용례 검색:** 이 문제와 유사한 `examples`를 찾으려면 어떤 구체적인 단어나 구절로 검색해야 할까?
EOF

python -m run.test\
    --output "믿음아오류없이해라.json"\
    --model_id "K-intelligence/Midm-2.0-Base-Instruct"\
    --model_ckpt_path ""\
    --device cuda:0\
    --batch_size 1\
    --prompt_type "generate_search_queries"\
    --prompt "$SYSTEM_PROMPT"\
    --unsloth None

python -m run.hybrid_test_queryex

