import argparse
import json
import tqdm
import os
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
from src.data import CustomDataset
from src.evaluation import extract_think_and_answer, extract_answer_and_reason, normalize_answer_text, calc_exact_match, calc_f1_score, calc_ROUGE_1, calc_bertscore

# print(torch.__version__, torch.version.cuda) 2.7.1+cu126 12.6

# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--model_path", type=str, default=None, help="huggingface model id")
g.add_argument("--is_quant", action="store_true", help="Enable quantinization")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
# fmt: on


def main(args):
    # Prepare model loading kwargs
    # if args.use_auth_token:
    #     model_kwargs["use_auth_token"] = args.use_auth_token
    
    if args.is_quant:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,             # 8-bit면 load_in_8bit=True
            bnb_4bit_quant_type="nf4",     # fp4·nf4 중 선택
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"]
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # **model_kwargs,
        quantization_config=bnb_cfg if args.is_quant else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir='model_cache',
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    if args.model_path is not None:
        model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    # Prepare tokenizer loading kwargs
    tokenizer_kwargs = {}
    if args.use_auth_token:
        tokenizer_kwargs["use_auth_token"] = args.use_auth_token
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    file_test = args.input
    dataset = CustomDataset(file_test, tokenizer)

    with open(file_test, "r") as f:
        result = json.load(f)

    print("### start")
    ems, rouge1s, bertscores = [], [], []
    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        input_ids = inp['input_ids']
        context_length = inp['input_length']
        print("### Prompt:\n" + tokenizer.decode(input_ids[:context_length], skip_special_tokens=False).strip())
        outputs = model.generate(
            input_ids[:context_length].unsqueeze(0).cuda(),
            # streamer=streamer,
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.8,
            # do_sample=True,
            # num_beams=3, 
            # num_return_sequences=5,
            use_cache=True
        )

        # output_texts = []
        # for i in range(outputs.size(0)):
        # print(outputs.shape, inp.shape[-1])
        # print(tokenizer.decode(outputs[inp.shape[-1]:], skip_special_tokens=False))
        output_text = tokenizer.decode(outputs[0][context_length:-1], skip_special_tokens=True).strip()
        print("output:", output_text)
        answer_text = tokenizer.decode(input_ids[context_length:-1], skip_special_tokens=False).strip()
        print("answer:", answer_text)

        # output_think, output_text = extract_think_and_answer(output_text)


        # print("[THINK]")
        # print(output_think)
        # print()
        # print("[ANSWER]")
        # print(output_text)


        # input()
        pred_ans, pred_reason = extract_answer_and_reason(output_text)
        pred_ans = normalize_answer_text(pred_ans)
        true_ans, true_reason = extract_answer_and_reason(answer_text)
        true_ans = normalize_answer_text(true_ans)

        em = calc_exact_match([true_ans], [pred_ans])
        # rouge_1 = calc_ROUGE_1([true_reason], [pred_reason])
        bertscore = calc_bertscore([true_reason], [pred_reason])

        ems.append(em)
        # rouge1s.append(rouge_1)
        bertscores.append(bertscore)
        
        # 출력에서 "답변: " 접두어 제거
        if output_text.startswith("답변: "):
            output_text = output_text[4:]
        elif output_text.startswith("답변:"):
            output_text = output_text[3:]

            # output_texts.append(output_text)
        
        result[idx].pop("retrieval")
        result[idx]["output"] = {"answer": output_text}
        # result[idx]["output"].update({"think": output_think})
        # result[idx]["output"].update({"prediction": output_text})
        # result[idx]["output"].update({"evaluation": {"em": em,
        #                                              "rouge1": rouge_1,
        #                                              "bertscore": bertscore}})

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

    print("EM:", sum(ems)/len(ems))
    # print("ROUGE_1:", sum(rouge1s)/len(rouge1s))
    print("BERTScore:", sum(bertscores)/len(bertscores))

    with open(f'{args.output[:-5]}_metrics.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps({"EM":sum(ems)/len(ems),
                            # "ROUGE_1": sum(rouge1s)/len(rouge1s),
                            "BERTScore": sum(bertscores)/len(bertscores)
                            }))


if __name__ == "__main__":
    exit(main(parser.parse_args()))