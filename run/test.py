
import argparse
import json
from tqdm import tqdm
# from unsloth import FastLanguageModel
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,GenerationConfig
from src.utils import set_random_seed

from src.data_original import CustomDataset, DataCollatorForSupervisedDataset, DataCollatorForInferenceDataset
from peft import PeftModel, PeftConfig
import os


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--model_ckpt_path", type=str, required=True, help="model checkpoint path")
g.add_argument("--batch_size", type=int, default=2, help="batch size")
g.add_argument("--prompt_type", type=str, default='mode_with_special_tokens', help="prompt type")
g.add_argument("--prompt", type=str, default="You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.", help="prompt")
g.add_argument("--unsloth", type=str, default="False", help="Path to model checkpoint to load.")

# fmt: on

def main(args):
    
    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set padding_side to left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    try:
        dataset = CustomDataset("../data/korean_language_rag_V1.0_test.json",
                                #  "/home/nlplab/ssd2/jeong/RAG/GCU-koreanRAG/data/retrieved_chunks_test.json",
                                "../result/reranking_results_믿오없.json",
                                "../data/few_shot_test_0.3.json",
                                tokenizer, 
                                args.prompt,
                                args.prompt_type)
    except Exception as e:
        dataset = CustomDataset("../data/korean_language_rag_V1.0_test.json",
                                #  "/home/nlplab/ssd2/jeong/RAG/GCU-koreanRAG/data/retrieved_chunks_test.json",
                                None,
                                "../data/few_shot_test_0.3.json",
                                tokenizer, 
                                args.prompt,
                                args.prompt_type)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForInferenceDataset(tokenizer),
    )
    
    if args.unsloth == "True":
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     model_name=args.model_id,
        #     max_seq_length=2048,
        #     dtype=torch.bfloat16,
        #     load_in_4bit=True,
        #     trust_remote_code=True,
        # )
        # tokenizer.padding_side = "left"
        # tokenizer.pad_token = tokenizer.eos_token
        # model = PeftModel.from_pretrained(model, args.model_ckpt_path)
        # FastLanguageModel.for_inference(model)
        None
    elif args.unsloth == "False":
        config = PeftConfig.from_pretrained(args.model_ckpt_path)
        quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_ckpt_path)

    else:
        if "Qwen" in args.model_id:
            model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
            )
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        else:
            quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                        )

            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                #quantization_config=quantization_config,
                trust_remote_code=True
            )
            
    try:
        generation_config = GenerationConfig.from_pretrained(args.model_id)
    except:
        None

    # FastLanguageModel.for_inference(model)
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)

    with open("../data/korean_language_rag_V1.0_test.json", "r") as f:
        result = json.load(f)
    print(len(result))
    batch_start_idx = 0
    for batch in tqdm(test_dataloader, desc="Test"):
        inp = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        outputs = model.generate(
            inp,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            generation_config=generation_config,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size = 20,
            do_sample=False,
            early_stopping=True,
            repetition_penalty=1.0
        )

        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output[inp.shape[-1]:], skip_special_tokens=False)
            generated_texts.append(text)

        # Replace special tokens with speaker IDs
        for i, text in enumerate(generated_texts):
            text = text.replace("[|endofturn|]", "")
            text = text.replace("<|end_of_text|>", "")
            text = text.replace("<|begin_of_text|>", "")
            text = text.replace("\n", "")
            text = text.replace("assistant", "")
            text = text.replace("<|im_end|>", "")
            text = text.replace("<|im_start|>", "")
            text = text.replace("<|start_header_id|>", "")
            text = text.replace("<|end_header_id|>", "")
            text = text.replace("<|start_of_text|>", "")
            text = text.replace("<|end_of_text|>", "")
            text = text.replace("<|eot_id|>", "")            
            text = text.replace("<s>", "")
            text = text.replace("한국어 문법 전문가입니다. 주어진 규칙을 참고하여 문제에 답변해 드리겠습니다.", "")

            result[batch_start_idx + i]["output"] = {"answer": text}
            print(result[batch_start_idx + i]["output"])

        batch_start_idx += len(generated_texts)

    with open(f"../result/{args.output}", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))