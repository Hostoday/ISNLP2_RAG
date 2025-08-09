import argparse

from datetime import datetime, timezone, timedelta

def get_args():
    parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")
    
    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--model_id", type=str, required=True, help="model file path")
    g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
    g.add_argument("--device", type=str, required=True, help="device to load the model")
    g.add_argument("--epoch", type=int, default=5, help="training epoch")
    g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    g.add_argument("--gradient_accumulation_steps", type=int, default=10, help="gradient accumulation steps")
    g.add_argument("--save_dir", type=str, default="resource/model", help="model save path")
    g.add_argument("--batch_size", type=int, default=1, help="batch size")
    g.add_argument("--seed", type=int, default=42, help="random seed")
    g.add_argument("--scheduler_type", type=str, default="cosine", help="scheduler type")
    g.add_argument("--warmup_steps", type=int, default=20, help="warmup steps")
    g.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    g.add_argument("--save_total_limit", type=int, default=5, help="save total limit")
    g.add_argument("--max_seq_len", type=int, default=1024, help="max sequence length")
    g.add_argument("--save_steps", type=int, default=50, help="max sequence length")
    g.add_argument("--eval_steps", type=int, default=50, help="evaluation steps")
    g.add_argument("--do_eval", type=str, default="True", help="evaluation flag")
    g.add_argument("--quantization", type=str, default="True", help="evaluation flag")
    g.add_argument("--prompt_type", type=str, default='format_example', help="prompt type")
    g.add_argument("--prompt", type=str, default="You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.", help="prompt")
    g.add_argument("--model_ckpt_path", type=str, default=None, help="Path to model checkpoint to load.")
    g.add_argument("--unsloth", type=bool, default=False, help="Path to model checkpoint to load.")

    g = parser.add_argument_group("Quantization Parameter")
    g.add_argument("--4bit", action="store_true", help="4bit quantization flag")
    
    g.add_argument("--lora_r", type=int, default=16, help="lora r value")
    g.add_argument("--lora_alpha", type=int, default=16, help="lora alpha value")
    g.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout value")
    
    g = parser.add_argument_group("Wandb Options")
    g.add_argument("--wandb", type=str,default="True")
    g.add_argument("--wandb_run_name", type=str, default=f'{datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%H-%M")}', help="wandb run name")
    g.add_argument("--wandb_project_name", type=str, default="RAG_Korean", help="wandb project name")
    g.add_argument("--wandb_entity", type=str, default="wjdghwns1096", help="wandb entity name")
    
    g = parser.add_argument_group("Inference Strategy")
    g.add_argument("--num_beams", type=int, default=1, help="num beams")
    g.add_argument("--top_k", type=int, default=50, help="top k")
    g.add_argument("--top_p", type=float, default=1.0, help="top p")
    
    g.add_argument("--trainer", type=str, required=True, help="Trainer usage")
    
    
    args = parser.parse_args()
    return args
