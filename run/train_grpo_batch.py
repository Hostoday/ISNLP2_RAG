import json, os, argparse, math, pathlib, logging, sys, wandb

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, get_scheduler
from peft import LoraConfig, TaskType, get_peft_model
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.evaluation import extract_answer_and_reason, normalize_answer_text, calc_exact_match, calc_ROUGE_1, calc_bertscore#, calc_bleurt

parser = argparse.ArgumentParser(prog="train", description="Training about Korean RAG.")

parser.add_argument("--train_input", type=str, required=True, help="input filename")
parser.add_argument("--eval_input", type=str, help="eval input filename")
parser.add_argument("--output", type=str, required=False, help="output filename")
parser.add_argument("--model_id", type=str, required=True, help="huggingface model id")
parser.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
parser.add_argument("--device", type=str, required=True, help="device to load the model")
parser.add_argument("--wandb_run_name", type=str, required=True, help="device to load the model")

# ── generation 하이퍼파라미터 ──
parser.add_argument("--max_new_tokens", type=int, default=1024, help="generate 시 생성 토큰 길이 한계")
parser.add_argument("--repetition_penalty", type=float, default=1.1)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.8)
parser.add_argument("--num_candidates", type=int, default=5)
parser.add_argument("--diversity_penalty", type=float, default=0.7)

# ── LoRA 설정 ──
parser.add_argument("--rank", type=int, default=8, help="LoRA 저랭크 차원 r")
parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA scaling factor α")

# ── 학습 스케줄 / 배치 ──
parser.add_argument("--batch_size", type=int, default=4, help="학습·평가 batch 크기")
parser.add_argument("--epochs", type=int, default=1, help="학습 epoch (소수 허용)")
parser.add_argument("--lr", type=float, default=2e-5, help="초기 learning-rate")
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="total_steps 대비 warm-up 비율(0-1)")
parser.add_argument("--gradient_ckpt", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--train", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--eval", action="store_true", help="Enable gradient checkpointing")
# ── 보상 스케쥴 ──
parser.add_argument("--w_em", type=float, default=0.25, help="total_steps 대비 warm-up 비율(0-1)")
parser.add_argument("--w_f1", type=float, default=0.25, help="total_steps 대비 warm-up 비율(0-1)")
parser.add_argument("--w_r1", type=float, default=0.25, help="total_steps 대비 warm-up 비율(0-1)")
parser.add_argument("--w_bs", type=float, default=0.25, help="total_steps 대비 warm-up 비율(0-1)")

parser.add_argument("--w_grpo", type=float, default=0.33, help="total_steps 대비 warm-up 비율(0-1)")
parser.add_argument("--w_ent", type=float, default=0.33, help="total_steps 대비 warm-up 비율(0-1)")
parser.add_argument("--w_ce", type=float, default=0.33, help="total_steps 대비 warm-up 비율(0-1)")


class GPROModels(torch.nn.Module):
    def __init__(self, tokenizer, model, args, generation_config, logger):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.generation_config = generation_config
        self.IGNORE_INDEX = -100
        self.w_em = args.w_em
        self.w_f1 = args.w_f1
        self.w_r1 = args.w_r1
        self.w_bs = args.w_bs
        self.logger = logger

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        true_labels = inputs['labels']
        context_length = inputs["input_length"]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids[:,:context_length],
                                          do_sample=False,
                                          num_beams=self.args.num_candidates,
                                          num_beam_groups=self.args.num_candidates,
                                          diversity_penalty=self.args.diversity_penalty,
                                          num_return_sequences=self.args.num_candidates, 
                                          generation_config=self.generation_config)
        # end_indices = [(output==self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0] for output in outputs]
        # candidates = [self.tokenizer.decode(output[context_length:end_idx]) for output, end_idx in zip(outputs, end_indices)]
        candidates = [self.tokenizer.decode(output[context_length:], skip_special_tokens=True) for output in outputs]
        self.logger.info("\n" + "\n\n".join([f'[{i+1}th candidate]:\n'+ candidate for i, candidate in enumerate(candidates)]) + "\n")
        # candidates = ["\"그리고 임금님의 행차 앞에서 어려운 사정을 아뢰어 다행히도 주민들의 삶을 보전하는 혜택을 입었다.\"가 옳다. '-이'와 '-히'로 끝나는 부사를 구분하는 방법은 [이]로만 소리가 나면 '-이'로 적고 [히]로도 소리가 나면 '-히'로 적는 것인데, 실제로는 발음을 잘 모르는 경우가 많기 때문에 발음을 기준으로는 구별하기가 어렵다. '-히'는 주로 '-하다'가 붙는 어근 뒤에서 나타나지만 '다행히'처럼 이런 경향성으로도 완전히 구별할 수 없는 경우가 있으므로 단어마다 국어사전을 확인하는 것이 좋다.", 
        #               "\"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다.",
        #               "\"우동이 불을 것 같아 걱정이다.\"가 옳다. '붇다'의 어간 끝 받침 'ㄷ'은 모음으로 시작하는 어미 앞에서 'ㄹ'로 바뀐다. 따라서 '붇다'에 관형형 어미 '-(으)ㄹ'이 결합하면 '불을'이 된다. 마찬가지로 '깨닫다'는 '깨달을', '듣다'는 '들을'으로 활용한다. 다만, '곧다', '뜯다' 등은 '곧을', '뜯을'과 같이 어간이 바뀌지 않는 형태로 활용한다."]
        gold_answer = self.tokenizer.decode(true_labels[true_labels!=-100], skip_special_tokens=True)
        self.logger.info(f"\n[gold answer]: {gold_answer}\n")
        
        gold_ans, gold_reason = extract_answer_and_reason(gold_answer)
        gold_ans = normalize_answer_text(gold_ans)

        cand_answers = []
        cand_reasons = []
        for cand in candidates:
            cand_ans, cand_reason = extract_answer_and_reason(cand)
            cand_ans = normalize_answer_text(cand_ans)
            cand_answers.append(cand_ans)
            cand_reasons.append(cand_reason)
        
        em = [calc_exact_match([gold_ans], [cand_ans]) for cand_ans in cand_answers]
        # rouge_L = [calc_ROUGE_L([gold_ans], [cand_ans]) for cand_ans in cand_answers]
        rouge_1 = [calc_ROUGE_1([gold_reason], [cand_reason]) for cand_reason in cand_reasons]
        bertscore = [calc_bertscore([gold_reason], [cand_reason]) for cand_reason in cand_reasons]

        # bleurt = [calc_bleurt([gold_reason], [cand_reason]) for cand_reason in cand_reasons]
        # des_avg = [(r1 + bs + bl) / 3 for r1, bs, bl in zip(rouge_1, bertscore, bleurt)]
        # final_score = [(em + des_avg)/2 for e, d_avb in zip(em, des_avg)]
        
        # rewards = torch.tensor([self.w_em*e + self.w_f1*rl + 5*self.w_r1*r1 + 5*self.w_bs*bs for e,rl,r1,bs in zip(em, rouge_L, rouge_1, bertscore)], device=outputs.device)
        rewards = torch.tensor([self.w_em*e + self.args.num_candidates*self.w_r1*r + self.args.num_candidates*self.w_bs*bs for e,r,bs in zip(em, rouge_1, bertscore)], device=outputs.device)
        # rewards = torch.tensor(final_score, device=outputs.device)
        reward_mean = rewards.mean()
        advantages = rewards - reward_mean

        # candidates_ids = [torch.concat((input_ids[:context_length], output[context_length:end_idx].unsqueeze(0)), dim=-1) for output, end_idx in zip(outputs, end_indices)]
        # candidates_labels = [torch.concat((torch.LongTensor([self.IGNORE_INDEX] * input_ids[:context_length].shape[-1]).to(outputs.device), output[context_length:end_idx])) for output, end_idx in zip(outputs, end_indices)]
        candidates_ids = [torch.concat((input_ids[:,:context_length], output[context_length:].unsqueeze(0)), dim=-1) for output in outputs]
        candidates_labels = [torch.concat((torch.LongTensor([self.IGNORE_INDEX] * input_ids[:,:context_length].shape[-1]).to(outputs.device), output[context_length:])) for output in outputs]
        
        cand_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ids.squeeze(0) for ids in candidates_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).unsqueeze(1)
        cand_attn_mask = cand_input_ids.ne(self.tokenizer.pad_token_id).long()
        cand_labels = torch.nn.utils.rnn.pad_sequence([lbls for lbls in candidates_labels], batch_first=True, padding_value=-100)

        self.model.train()
        # grpo_losses = []
        # for i, (c_inputs, c_masks, c_labels) in enumerate(zip(cand_input_ids, cand_attn_mask, cand_labels)):
        # print(cand_input_ids.shape, cand_attn_mask.shape, cand_labels.shape)
        output = self.model(cand_input_ids.squeeze(), 
                            attention_mask=cand_attn_mask.squeeze(), 
                            labels=cand_labels)
            
        target_ids = cand_input_ids[:,:,context_length+1:]
        # log_probs = torch.log_softmax(output.logits[:,context_length:-1,:], dim=-1)
        token_probs = torch.nn.functional.softmax(output.logits[:,context_length:-1,:], dim=-1)
        # token_logp = log_probs.gather(2, target_ids.unsqueeze(-1).long()).squeeze(-1)
        
        # print(target_ids.shape, token_probs.shape, output.logits.shape)
        # torch.Size([5, 1, 135]) torch.Size([5, 135, 131384]) torch.Size([5, 1333, 131384])
        token_probs = token_probs.gather(2, target_ids.squeeze(1).unsqueeze(-1).long()).squeeze(-1)
        
        # mask = torch.isfinite(token_logp)
        # token_logp = token_logp[mask]
        # token_probs = torch.exp(token_logp[mask])

        # ratio = torch.exp(token_logp - token_logp.detach())
        ratio = torch.exp(torch.log(token_probs + 1e-10) - torch.log(token_probs.detach() + 1e-10))
        # print(ratio.shape, advantages.shape)
        # torch.Size([5, 135]) torch.Size([5]) 
        eps_clip = 0.1
        # grpo_loss = -torch.mean(torch.min(ratio*advantages.unsqueeze(-1), torch.clamp(ratio, 1-eps_clip, 1+eps_clip)*advantages[i]))
        grpo_loss = -torch.mean(ratio*advantages.unsqueeze(-1))
        # grpo_losses.append(grpo_loss)
        
        # grpo_loss = sum(grpo_losses) / len(outputs)

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        output = self.model(input_ids, 
                            attention_mask=attn_mask, 
                            labels=true_labels)
        ce_loss = output.loss

        target_ids = input_ids[:,context_length+1:]
        log_probs = torch.log_softmax(output.logits[:,context_length:-1,:], dim=-1)
        token_logp = log_probs.gather(2, target_ids.unsqueeze(-1).long()).squeeze(-1)
        mask = torch.isfinite(token_logp)
        token_logp = token_logp[mask]
        token_probs = torch.exp(token_logp)
        entropy_loss = -torch.mean(token_probs*token_logp)

        eval = {"em": sum(em)/len(em), 
                # "rouge_L": sum(rouge_L)/len(rouge_L), 
                "rouge_1": sum(rouge_1)/len(rouge_1), 
                "bertscore": sum(bertscore)/len(bertscore),
                # "bleurt": sum(bleurt)/len(bleurt),
                "reward_mean": reward_mean.item()}
        
        loss = {"grpo_loss":grpo_loss, 
                "entropy_loss":entropy_loss, 
                "ce_loss":ce_loss}
        
        self.logger.info("\nEval")
        for k in eval:
            self.logger.info(f"{k}: {eval[k]}")
        self.logger.info("\nLoss")
        for k in loss:
            
            self.logger.info(f"{k}: {loss[k].item()}")

        return (loss, eval)

def main(args):
    out_dir = pathlib.Path(args.output or "./grpo_ckpt")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = pathlib.Path(out_dir) / "log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),    # 그대로 터미널에도 출력
        ],
    )
    logger = logging.getLogger(__name__)
    args_dict = vars(args)

    logger.info("Args")
    for k in args_dict:
        logger.info(f"{k}: {args_dict[k]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    ]

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                  eos_token_id=terminators,
                                  pad_token_id=tokenizer.eos_token_id,
                                  repetition_penalty=args.repetition_penalty,
                                  temperature=args.temperature,
                                  top_p=args.top_p,
                                  early_stopping=True,
                                  use_cache=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,             # 8-bit면 load_in_8bit=True
        bnb_4bit_quant_type="nf4",     # fp4·nf4 중 선택
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=["lm_head"]
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                 quantization_config=bnb_cfg, 
                                                 torch_dtype=torch.bfloat16, 
                                                 device_map="auto", 
                                                 cache_dir='model_cache', 
                                                #  attn_implementation="flash_attention_2", 
                                                 trust_remote_code=True)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             r=args.rank,
                             lora_alpha=args.lora_alpha,
                             target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                             lora_dropout=0.1)

    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    if args.gradient_ckpt:
        model.gradient_checkpointing_enable()
    model = GPROModels(tokenizer, model, args, gen_config, logger)

    if wandb.run is None:
        wandb.init(
            project="Korean_QA_RAG",
            name=f"{args.model_id.split('/')[-1]}_finetune_{args.wandb_run_name}",
            config=vars(args)
        )

    # with open(args.input, 'r') as f:
    #     train_data = json.load(f)
    with open(args.eval_input, 'r') as f:
        eval_data = json.load(f)

    train_dataset = CustomDataset(args.train_input, tokenizer)
    eval_dataset = CustomDataset(args.eval_input, tokenizer)
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_training_steps = math.ceil(len(train_dataloader) * args.epochs)
    scheduler = get_scheduler("linear", 
                               optimizer,
                               num_warmup_steps=int(total_training_steps*args.warmup_ratio),
                               num_training_steps=total_training_steps)
    
    if args.train:
        model.train()
        for e, epoch in enumerate(range(int(args.epochs))):
            logger.info(f"\nStart epoch {epoch+1}/{args.epochs}")
            with tqdm(train_dataloader, unit='batch') as train_batch:
                for b, batch in enumerate(train_batch):
                    train_batch.set_description('Train')
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].cuda()
                    logger.info("\n### PROMPT:\n" + tokenizer.decode(batch["input_ids"][0][:batch["input_length"]]))
                    
                    loss, eval = model(batch)
                    grpo_loss, entropy_loss, ce_loss = loss["grpo_loss"], loss["entropy_loss"], loss["ce_loss"]
                    total_loss = args.w_grpo*grpo_loss + args.w_ent*entropy_loss + args.w_ce*ce_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    train_batch.set_postfix(total_loss=total_loss.item(),
                                            grpo_loss=grpo_loss.item(),
                                            entropy_loss=entropy_loss.item(),
                                            ce_loss=ce_loss.item(),
                                            em=eval["em"],
                                            # rl=eval["rouge_L"],
                                            r1=eval["rouge_1"],
                                            bs=eval["bertscore"],
                                            # bl=eval["bleurt"],
                                            rewards=eval["reward_mean"],
                                            lr=optimizer.param_groups[0]['lr'])
                    
                    if wandb.run is not None:    
                        wandb.log({
                            "train/loss":               total_loss.item(),
                            "train/loss/grpo":          grpo_loss.item(),
                            "train/loss/entropy":       entropy_loss.item(),
                            "train/loss/ce":            ce_loss.item(),
                            "train/rewards/em":         eval["em"],
                            # "train/rewards/rougeL":     eval["rouge_L"],
                            "train/rewards/rouge1":     eval["rouge_1"],
                            "train/rewards/bertscore":  eval["bertscore"],
                            # "train/rewards/bleurt":     eval["bleurt"],
                            "train/rewards":            eval["reward_mean"],
                            "train/learning_rate":      optimizer.param_groups[0]['lr'],
                            "train/epoch":              epoch,
                            "train/global_step":        e*len(train_dataloader)+b
                        })

            save_path = out_dir / f"epoch_{epoch+1}"
            save_path.mkdir(exist_ok=True)
            model.model.save_pretrained(save_path)
        
            if args.eval:
                model.eval()
                eval_metrics = {"em": [], "r1": [], "bs": []}
                predictions = []
                with torch.no_grad():
                    with tqdm(eval_dataloader, unit='batch') as eval_batch:
                        for i, batch in enumerate(eval_batch):
                            eval_batch.set_description('Eval')
                            for k in batch:
                                if isinstance(batch[k], torch.Tensor):
                                    batch[k] = batch[k].to(args.device)
                            input_ids = batch['input_ids']
                            true_labels = batch['labels']
                            context_length = batch["input_length"]

                            output = model.model.generate(input_ids[:,:context_length],
                                                        generation_config=gen_config)
                            pred_answer = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
                            gold_answer = tokenizer.decode(true_labels[true_labels!=-100], skip_special_tokens=True)
                            
                            predictions.append(pred_answer)
                            logger.info("\n[prediction]:", pred_answer)
                            logger.info("\n[gold]:", gold_answer)

                            gold_ans, gold_reason = extract_answer_and_reason(gold_answer)
                            gold_ans = normalize_answer_text(gold_ans)

                            pred_ans, pred_reason = extract_answer_and_reason(pred_answer)
                            pred_ans = normalize_answer_text(pred_ans)
                            
                            em = calc_exact_match([gold_ans], [pred_ans])
                            # rL = calc_ROUGE_L([gold_ans], [pred_ans])
                            rouge_1 = calc_ROUGE_1([gold_reason], [pred_reason])
                            bertscore = calc_bertscore([gold_reason], [pred_reason])
                            # bleurt = calc_bleurt([gold_reason], [pred_reason])

                            eval_metrics["em"].append(em)
                            # eval_metrics["rL"].append(rL)
                            eval_metrics["r1"].append(rouge_1)
                            eval_metrics["bs"].append(bertscore)
                            # logger.info(f"\n[Eval {i+1}th sample]: {eval_metrics}")
                            logger.info(f"\n[Eval {i+1}th sample]:")
                            for k in eval_metrics:
                                logger.info(f'{k}: {eval_metrics[k][-1]:.4f}')
                            # eval_metrics["bl"] = bleurt
                            # eval_metrics["des_avg"] = (rouge_1 + bertscore + bleurt) / 3
                            # eval_metrics["final_score"] = (em + eval_metrics["descriptive_avg"]) / 2

                            eval_batch.set_postfix(em=em,
                                                #    rL=rL,
                                                r1=rouge_1,
                                                bs=bertscore,
                                                # bl=bleurt,
                                                # des_avg=eval_metrics["des_avg"],
                                                # final_score=eval_metrics["final_score"])
                            )
                n_eval = len(eval_dataloader)
                logger.info(f"\n[Eval] epoch {epoch+1}\n")
                for k in eval_metrics:
                    logger.info(f'{k}: {sum(eval_metrics[k])/n_eval:.4f}')
                
                if wandb.run is not None:
                    wandb.log({"eval/em":          sum(eval_metrics["em"])/n_eval,
                            #    "eval/rL":          sum(eval_metrics["rL"])/n_eval,
                            "eval/rouge1":      sum(eval_metrics["r1"])/n_eval,
                            "eval/bertscore":   sum(eval_metrics["bs"])/n_eval,
                            #   "eval/bleurt":      eval_metrics["bl"],
                            #   "eval/des_avg":     eval_metrics["des_avg"],
                            #   "eval/final_score": eval_metrics["final_score"],
                            #    "epoch":            epoch+1,
                            })

            for ed, prediction in zip(eval_data, predictions):
                ed["output"].update({"prediction": prediction})
            
            with open(f"{args.output}/{args.model_id.split('/')[-1]}_finetune_{args.wandb_run_name}_pred_ep{epoch}.json", 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, indent=4, ensure_ascii=False)

    final_path = out_dir / "final"
    final_path.mkdir(exist_ok=True)
    model.model.save_pretrained(final_path)
            
if __name__ == '__main__':
    exit(main(parser.parse_args()))
    
