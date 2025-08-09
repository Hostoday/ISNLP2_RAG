import torch
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from src.data import CustomDataset, DataCollatorForSupervisedDataset, DataCollatorForInferenceDataset
from src.utils import set_random_seed
from peft import PeftModel, PeftConfig
from src.arg_parser import get_args
from korouge_score import rouge_scorer
import argparse
import os

def rouge_score(prediction, reference):
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    avg_score = 0
    for pred, ref in zip(prediction, reference):
        # Calculate the ROUGE score for each pair of prediction and reference
        scores = rouge.score(pred, ref)
        avg_score += scores['rouge1'].fmeasure
    return {"rouge1": avg_score / len(prediction)}

def get_label_prefix_mask_length(label):
    for i, val in enumerate(label):
        if val != -100:
            return i
    return len(label)

def compute_metrics(predictions, labels):
    rouge_1 = rouge_score(predictions, labels)
    pred_parts = [p.split("\"") for p in predictions]
    label_parts = [l.split("\"") for l in labels]
    count = 0
    total = 0
    for p, l in zip(pred_parts, label_parts):
        total += 1
        if len(p) > 1 and p[1] == l[1]:
            count += 1
    accuracy = count / total if total > 0 else 0.0

    return {"EM": accuracy, "rouge1": rouge_1["rouge1"]}


def unsloth_init(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(model, args.model_ckpt_path)
    FastLanguageModel.for_inference(model)

    return model, tokenizer

def init_model(args):
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=args.device,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    config = PeftConfig.from_pretrained(args.model_ckpt_path)
    model = PeftModel.from_pretrained(model, args.model_ckpt_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main(args):
    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set padding_side to left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = CustomDataset(
            "/home/nlplab/ssd1/hyunmin/data/korean_language_rag_V1.0_dev.json",
            "/home/nlplab/ssd1/hyunmin/data/retrieved_chunks_hybrid_dev_0.3.json",
            tokenizer,
            args.prompt
        )
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForInferenceDataset(tokenizer),
    )
    if args.unsloth:
        model, tokenizer = unsloth_init(args)
    else:
        model, tokenizer = init_model(args)

    model.eval()
    model.to(args.device)
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512, eos_token_id=terminators)
            outputs = outputs[:, input_ids.shape[1]:]
            outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(outputs_decoded)
    evaluation_results = compute_metrics(results, dataset.label_text)
    print(f"Evaluation Results: {evaluation_results}")
    model_name_safe = args.model_id.replace("/", "__")  # 또는 "_" 등
    save_path = f"inference/evaluation/{model_name_safe}_evaluation.json"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))     

if __name__ == "__main__":
    args = get_args()
    exit(main(args))