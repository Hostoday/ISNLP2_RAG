import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, fname, rfname, ffname, tokenizer, prompt,prompt_type="format_example"):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.label_text = []
        
        PROMPT = prompt
        with open(fname, encoding="utf-8") as f:
            data = json.load(f)
        if rfname is None:
            rfname = [" "]*len(data)
        else:
            with open(rfname, encoding="utf-8")as rf:
                rdata = json.load(rf)

        with open(ffname, encoding="utf-8")as ff:
            fdata = json.load(ff)
            

        def format_example(example,retrieve_doc):
            question = example["input"]["question"]
            context = retrieve_doc["retrieved_chunks"]
            context_total = []
            print(context[0]["text"].replace("\n",""), context[0]["examples"])
            for i in range(len(context)):
                context_total.append(context[i]["text"].replace("\n","")+"\n"+context[0]["examples"])
            message = f"주어진 문제를 문서를 기반으로 답변을 작성해주세요. 문제: {question} \n 문서1: {context[0]}\n 문서2: {context[1]}\n 문서3: {context[2]}\n 답변: "

            return message
        
        def few_example(example,retrieve_doc,few_doc):
            question = example["input"]["question"]
            context = retrieve_doc["retrieved_chunks"]
            few_shot = few_doc["few_shots"]
            context_total = []
            few_shot_total = []
            few_shot_total.append(few_shot[0]["question"] + "\n"+few_shot[0]["answer"])

            for i in range(len(context)):
                context_total.append(context[i]["text"].replace("\n","")+"\n 예시:"+context[i]["examples"])
            request = "\n[요청사항] 주어진 문제를 문서를 기반으로 주어진 예시와 같은 형식의 답변을 작성해주세요.\n"
            context = f"문서1: {context_total[0]}\n 문서2: {context_total[1]}\n 문서3: {context_total[2]}"
            few_exam = "\n".join(few_shot_total)
            input_question = f"문제: {question} \n 답변: "
            message = request + few_exam + input_question
            return message

        def format_example_cot(example,retrieve_doc):
            question = example["input"]["question"]
            retrieved_chunks  = retrieve_doc["retrieved_chunks"]
            formatted_documents = []
            for i, chunk in enumerate(retrieved_chunks):
                text = chunk.get("text", "").replace("\n", " ")
                examples = chunk.get("examples", "")
                formatted_documents.append(f"문서 {i+1}: {text} (예: {examples})")
            documents_section = "\n".join(formatted_documents)
            message = f"""[지시] 당신은 한국어 문법 분석 전문가입니다. 아래 형식에 맞춰 단계별로 추론하고 최종 답변을 작성하세요.

                        ### 제공된 정보
                        {documents_section}

                        ### 분석할 문제
                        {question}

                        ---

                        ### 추론 과정 (단계별로 생각하세요)

                        1.  **문제 분석:** 질문에서 나타나는 문법적 오류의 핵심은 무엇인가?
                            - 
                        2.  **규칙 탐색 및 적용:** 제공된 문서들 중 이 오류를 설명하는 가장 적합한 규칙은 무엇이며, 어떻게 적용되는가?
                            - 
                        3.  **결론 도출:** 위 분석을 바탕으로, 왜 이 문장이 틀렸으며 올바른 표현은 무엇인지 명확히 결론 내리시오.
                            - 

                        ### 최종 답변
                        """
            return message
            
        def few_example_instruct_new(example, few_doc):
            question = example["input"]["question"].split("\n",maxsplit=1)
            if len(question)>1:
                sentence = question[1]
                sentence = sentence.replace("-","").strip()
                question = question[0]
            else:
                sentence = example["input"]["question"].split("\"")[1]
                sentence = sentence.replace("-","").strip()
                question = example["input"]["question"].split("\"")[0]
            type = example["input"]["question_type"]
            few_shots = few_doc["few_shots"]

            # 2. 예시: '입력'과 '출력' 라벨을 명확히 하여 패턴 학습 효과 극대화
            few_shot_texts = []
            for shot in few_shots:
                few_shot_texts.append(f"입력: {shot['question']}\n출력: {shot['answer']}")

            example_prompt = "### 예시\n" + "\n\n".join(few_shot_texts) + "\n"


            # 4. 최종 과업: 모델이 채워야 할 부분을 명확히 제시
            type_prompt = f"이 문제는 {type}의 문제입니다. \n"
            task_prompt = f"### 문제 \n{question}\n\n 문장\n{sentence} \n\n ### 최종답변:"

            # 최종 프롬프트 조합
            # 순서 변경: 지시사항 -> 예시 -> 근거 -> 과업 순으로 제시하여 학습 효과 증대
            final_prompt = type_prompt + example_prompt +  task_prompt
            # print(final_prompt)
            # input()
            return final_prompt


        def generate_search_queries(question_text,fewshot):
            """
            [Midm 맞춤형] 모델이 규정 문서에 최적화된 검색어를 생성하도록 유도하는 프롬프트를 생성합니다.
            """
            # [지침] 모델에게 수행할 작업을 명확하고 구조적으로 지시합니다.
            instruction = (
                "### 지시:\n"
                "아래 `### 문제`의 핵심 문법 쟁점을 분석하여, 관련 규정을 찾는 데 가장 효과적인 검색어 3개를 생성하십시오.\n"
                "검색어는 '조항 제목 검색', '규정 내용 검색', '구체적 용례 검색'의 세 가지 유형을 모두 고려해야 합니다.\n"
                "당신의 유일한 출력물은 `### 답변`에 명시된 JSON 형식이어야 합니다.\n\n"
            )

            # [예시] 모델이 따라야 할 명확한 입출력 패턴을 제공합니다.
            example_q1 = "\"아는 것이 힘이다\"에서 '것'의 띄어쓰기가 맞는지 설명하세요."
            example_a1 = '{"queries": ["띄어쓰기 - 한글 맞춤법 제42항", "의존 명사 띄어쓰기", "아는 것"]}'

            example_q2 = "\"불이 꺼져 간다\"에서 '꺼져 간다'는 붙여 써도 되나요?"
            example_a2 = '{"queries": ["띄어쓰기 - 한글 맞춤법 제47항", "보조 용언 붙여 씀 허용", "불이 꺼져간다"]}'

            # [NEW!] 애매한 질문에 대한 예외 처리 예시 추가
            example_q3 = "옛날에 쓰던 말인데 요즘엔 좀 다른 거 같은데 뭐가 맞나요?"
            example_a3 = '{"queries": ["옛날에 쓰던 말", "옛날에 쓰던 말인데 요즘엔 좀 다른 거 같은데 뭐가 맞나요?", "표준어 사정 원칙"]}'

            example_prompt = (
                "### 예시:\n\n"
                f"문제: {example_q1}\n답변: {example_a1}\n\n"
                f"문제: {example_q2}\n답변: {example_a2}\n\n"
                f"문제: {example_q3}\n답변: {example_a3}\n\n"
            )
            
            # [문제]와 [답변] 형식을 명확히 구분하여 모델의 역할을 지정합니다.
            task_prompt = "### 문제:\n" + question_text["input"]["question"] + "\n"
            output_format_prompt = "### 답변:\n"

            final_prompt = instruction + example_prompt + task_prompt + output_format_prompt
            # print(final_prompt)
            # input()
            return final_prompt
                
        def process_target_original(target, tokenizer):
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")
        
        for example, retrieve_doc,few_doc in zip(data, rdata, fdata):
            if prompt_type == "few_example":
                chat = few_example(example, retrieve_doc,few_doc)
            elif prompt_type == "few_example_instruct_new":
                chat = few_example_instruct_new(example,few_doc)
            elif prompt_type == "generate_search_queries":
                chat = generate_search_queries(example,few_doc)
            else:
                chat = format_example(example, retrieve_doc)
            process_target_func = process_target_original
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            try:
                source = tokenizer.apply_chat_template(
                    message,
                    tokenize = True,
                    force_reasoning=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            except:
                source = tokenizer.apply_chat_template(
                    message,
                    tokenize = True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                
            if "output" in example.keys():
                target = example["output"]["answer"]
            else:
                target = ""
            self.label_text.append(target)
            if target != "":
                target += tokenizer.eos_token
            else:
                target = tokenizer.eos_token

            target = process_target_func(target, tokenizer)
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

            
    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx], "labels": self.label[idx],"labels_text": self.label_text[idx]}

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        # print(self.tokenizer.decode(input_ids[0]))
        # print(self.tokenizer.decode(labels))
        return {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }

class DataCollatorForInferenceDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
