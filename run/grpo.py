import json, os, argparse, re, functools

parser = argparse.ArgumentParser(
    prog="train_grpo", description="GRPO + CE + Entropy Training"
)

# 필수 경로 / 모델
parser.add_argument("--input",   required=True, help="학습 JSON 파일")
parser.add_argument("--eval_input", type=str, default=None, help="검증용 JSON 파일 (없으면 평가 생략)")
parser.add_argument("--output",  default="grpo_ckpt", help="체크포인트 저장 경로")
parser.add_argument("--model_id",required=True, help="HF model id (base)")
parser.add_argument("--cache_dir", type=str, default=None, help="모델/토크나이저를 저장·로딩할 디렉터리")
# 보상 계수
parser.add_argument("--em_weight",    type=float, default=0.5)
parser.add_argument("--rouge_weight", type=float, default=0.5)
# 주요 손실 계수
parser.add_argument("--entropy_coef", type=float, default=0.01, help="λ_H (entropy bonus)")
parser.add_argument("--ce_coef",      type=float, default=0.2,  help="λ_CE (teacher-forcing CE)")
# ===== 학습 스케줄 관련 =====
parser.add_argument("--epochs",       type=float, default=1.0, help="학습 epoch 수 (float 도 허용, 1.5 epoch 등)")
parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=1e-6, help="초기 learning-rate")
parser.add_argument("--lr_scheduler_type", choices=["linear", "cosine", "constant", "cosine_with_restarts", "polynomial", "inverse_sqrt"], default="linear", help="HF 스케줄러 종류")
parser.add_argument("--warmup_steps",   type=int,   default=5000, help="warm-up 스텝 수(정수). 지정하면 warmup_ratio 무시")
parser.add_argument("--warmup_ratio",   type=float, default=0.1, help="total_steps 대비 warm-up 비율 (0-1)")
# 핵심 배치 · 샘플링
parser.add_argument("--batch_size",           type=int, default=4,   help="per-device train batch")
parser.add_argument("--steps_per_gen",        type=int, default=4,   help="steps_per_generation")
parser.add_argument("--num_generations",      type=int, default=2,   help="G completions / prompt")
parser.add_argument("--temperature",      type=float, default=0.7,   help="temperature")
parser.add_argument("--top_p",      type=float, default=0.9,   help="top_p")
parser.add_argument("--grad_accum",           type=int, default=4,   help="gradient_accumulation_steps")
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--gpus", type=str, default=0, help="사용할 GPU 인덱스(콤마 구분). 예: --gpus 0,1")
# 양자화 토글
parser.add_argument("--quant_bits", choices=[8,4,None], type=int, default=None, help="8 → load_in_8bit, 4 → load_in_4bit, None → FP16/BF16")
# 4-bit 고급 파라미터(선택)
parser.add_argument("--bnb_compute_dtype", choices=["fp16","bf16","fp32"], default="fp16")
parser.add_argument("--bnb_quant_type",    choices=["nf4","fp4"], default="nf4")
parser.add_argument("--bnb_use_double_quant", action="store_true", help="4-bit double-quant (QLoRA)")
# LoRA 켜기 / 하이퍼 파라미터
parser.add_argument("--use_lora",  action="store_true", help="LoRA 적용")
parser.add_argument("--lora_r",    type=int,   default=8)
parser.add_argument("--lora_alpha",type=float, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_target_modules", nargs="*", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], help="LoRA를 삽입할 모듈 이름 리스트")
# 길이·토큰 한계
parser.add_argument("--max_prompt_len",       type=int, default=512)
parser.add_argument("--max_completion_len",   type=int, default=128)
# PPO/GRPO 계수
parser.add_argument("--beta",           type=float, default=0.1,  help="KL penalty β")
parser.add_argument("--eps_low",        type=float, default=0.1,  help="clip ε_low")
parser.add_argument("--eps_high",       type=float, default=0.1,  help="clip ε_high")
parser.add_argument("--delta_cap",      type=float, default=None, help="ratio cap δ (optional)")
parser.add_argument("--num_iter",       type=int,   default=1,    help="μ (updates per generation)")
parser.add_argument("--loss_type",      choices=["grpo","bnpo","dr_grpo"], default="grpo")
# 기타 로깅·저장
parser.add_argument("--logging_steps",  type=int, default=10)
parser.add_argument("--save_steps",     type=int, default=1000)
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--report_to",      nargs="*", default=[], help="e.g. wandb")
# vLLM / Liger 토글
parser.add_argument("--use_liger",      action="store_true", help="fused Liger loss 사용")
parser.add_argument("--use_cot",      action="store_true", help="cot prompt 사용")
# wnadb logging
parser.add_argument("--wandb_project_name",  type=str, default="Korean_QA_RAG", help="wandb project name")
parser.add_argument("--wandb_run_name",  type=str, default="GRPO-midm", help="wandb run name")

args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer
from peft import LoraConfig, TaskType
import wandb

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args,
                 entropy_coef: float = 0.01,
                 ce_coef: float       = 0.2,
                 em_weight: float = 0.5,
                 rouge_weight: float = 0.5,
                 use_cot: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coef = entropy_coef
        self.ce_coef      = ce_coef
        self.em_weight     = em_weight
        self.rouge_weight  = rouge_weight
        self.use_cot       = use_cot
        # CE 계산에 쓸 토크나이저를 이미 self.processing_class 로 갖고 있음

    # -----------------------------------------------------------
    # (A)  reference token을 inputs 로 전달하기 위해
    #      _generate_and_score_completions 살짝 확장
    # -----------------------------------------------------------
    def _generate_and_score_completions(self, inputs):
        # base_outputs: dict( prompt_ids, prompt_mask, completion_ids, … )
        # print(inputs)
        base_outputs = super()._generate_and_score_completions(inputs)

        # -------- reference answer 토크나이즈 -------------------
        device = self.accelerator.device
        ref_texts = [ex["reference_answer"] for ex in inputs]
        ref_tok = self.processing_class(
            text=ref_texts, return_tensors="pt",
            padding=True, padding_side="right",
            add_special_tokens=False
        )
        ref_tok = {k: v.to(device) for k,v in ref_tok.items()}
        base_outputs["ref_ids"]   = ref_tok["input_ids"]
        base_outputs["ref_mask"]  = ref_tok["attention_mask"].int()
        return base_outputs

    # -----------------------------------------------------------
    # (B)  손실 계산: GRPO + entropy − λ_H + CE * λ_CE
    # -----------------------------------------------------------
    def _compute_loss(self, model, inputs):
        # 1) 기본 GRPO 손실
        grpo_loss = super()._compute_loss(model, inputs)

        # 2) --- 엔트로피 보너스 ---------------------------------
        prompt_ids       = inputs["prompt_ids"]
        completion_ids   = inputs["completion_ids"]
        completion_mask  = inputs["completion_mask"]
        input_ids_full   = torch.cat([prompt_ids, completion_ids], 1)
        attention_full   = torch.cat([inputs["prompt_mask"], completion_mask], 1)
        logits_to_keep   = completion_ids.size(1)

        logp = self._get_per_token_logps(
            model, input_ids_full, attention_full, logits_to_keep
        )                            # (B, Lc)
        p = torch.exp(logp)
        token_entropy = -(p * logp) * completion_mask
        entropy = token_entropy.sum() / completion_mask.sum().clamp(min=1)

        # 3) --- CE(teacher-forcing) on reference answer ---------
        ref_ids  = inputs["ref_ids"]
        ref_mask = inputs["ref_mask"]
        ref_input_ids  = torch.cat([prompt_ids, ref_ids], 1)
        ref_attn_mask  = torch.cat([inputs["prompt_mask"], ref_mask], 1)
        ref_logits_keep = ref_ids.size(1)

        ref_logp = self._get_per_token_logps(
            model, ref_input_ids, ref_attn_mask, ref_logits_keep
        )                            # (B, L_ref)

        # CE = − mean log p(ref_token)
        ce_nll = -(ref_logp * ref_mask).sum() / ref_mask.sum().clamp(min=1)

        # 4) --- 최종 손실 --------------------------------------
        loss = grpo_loss - self.entropy_coef * entropy + self.ce_coef * ce_nll

        # ---------- (2) 스칼라로 변환 ----------
        g  = grpo_loss.detach().float().mean().item()
        c  = ce_nll.detach().float().mean().item()
        ent= entropy.detach().float().mean().item()

        # ---------- (3) Trainer 메트릭 큐에 추가 ----------
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss/grpo"].append(g)
        self._metrics[mode]["loss/ce"].append(c)
        self._metrics[mode]["loss/entropy"].append(ent)

        return loss

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids):
        print("prompts:", prompts)
        # print("completions:", completions)
        # print("completion_ids:", completion_ids)
        device = self.accelerator.device
        B, G = len(inputs), len(completions[0])

        em_mat    = torch.zeros(B, G, device=device)
        rouge_mat = torch.zeros_like(em_mat)

        for i, (gt, gens) in enumerate(zip(
                [ex["reference_answer"] for ex in inputs], completions)):

            if self.use_cot:
                gt = _remove_cot(gt)

            gt_ans, gt_reason = _extract_answer_reason(gt)
            print("### gt_ans:", gt_ans, "\ngt_reason:", gt_reason)

            for j, gen in enumerate(gens):
                # ▶ 반드시 문자열로 변환
                pred_txt = _completion_to_text(gen)
                if self.use_cot:
                    pred_txt = _remove_cot(pred_txt)

                pr_ans, pr_reason = _extract_answer_reason(pred_txt)
                print("\n### pr_ans:", pr_ans, "\npr_reason:", pr_reason)

                em_mat[i, j]    = float(pr_ans.strip() == gt_ans.strip())
                rouge_mat[i, j] = _rouge1_f1(gt_reason, pr_reason)

        reward_mat = self.em_weight * em_mat + self.rouge_weight * rouge_mat

        # ---------- (3) Trainer 메트릭 큐에 추가 ----------
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["rewards/em"].append(em_mat.mean().item())
        self._metrics[mode]["rewards/rouge1"].append(rouge_mat.mean().item())

        return reward_mat

# ---------------------------------------------------------------------
# 1. Chat 프롬프트 템플릿 ------------------------------------------------
PROMPT_SYSTEM = (
    "You are a helpful AI assistant. Please answer the user's questions kindly. "
    "당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 "
    "AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. "
    "단, 동일한 문장을 절대 반복하지 마시오."
)

def make_chat(inp: dict) -> str:
    """JSON 한 항목(input)에 대해 user 프롬프트 문자열 생성"""
    type_instructions = {
        "선다형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
            "[예시]\n"
            "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
            "1) 주사위 놀이\n"
            "2) 검무\n"
            "3) 격구\n"
            "4) 영고\n"
            "5) 무애무\n"
            "답변: 3"
        ),
        "서술형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
            "[예시]\n"
            "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
            "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
            "[예시]\n"
            "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
            "답변: 정약용"
        ),
        "교정형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
            "[예시1]\n"
            "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
            "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다.\n"
            "[예시2]\n"
            "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"공부를 하던지 책을 읽던지 하고 싶은 걸 해라.\"\n"
            "답변: \"공부를 하든지 말든지 네 마음대로 해라.\"가 옳다. '-던'은 과거의 일을 전달할 때 사용하고 '-든'은 선택이나 무관의 뜻을 나타낼 때 사용한다. 마찬가지로 '-든'이 포함된 '-든지', '-든가' 등은 선택이나 무관의 뜻을, '-던'이 포함된 '-던지', '-던데' 등은 과거를 나타낸다. 이 문장은 공부를 하는 것과 마는 것 중 선택하라는 맥락을 담고 있으므로 '-든'을 써야 한다.\n"
            "[예시3]\n"
            "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"헷갈리게시리 굴지 마라.\"\n"
            "답변: \"헷갈리게끔 굴지 마라.\"가 옳다. 같은 의미로 사용되는 '-게끔'과 '-게시리' 중 '-게시리'는 꽤 많이 쓰이는 편이나 표준어에서 제외되었다. 방언형이기도 할뿐더러, 같은 의미의 어미 '-도록'이 널리 쓰이고 있어 '-게끔' 하나만 표준어로 삼아도 충분하다고 판단하였기 때문이다."
        ),
        "선택형": (
            "[예시1]\n"
            "질문: \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다.\n"
            "[예시2]\n"
            "질문: \"오늘은 날씨가 {푹하다/푸카다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "답변: \"오늘은 날씨가 푹하다.\"가 옳다. '-하다'나 '-없다'가 붙어서 된 용언은 그 '-하다'나 '-없다'를 밝히어 적는다. '푹하다'의 '푹'은 '노래하다'의 '노래'와 달리 자립적이지 않은 어근이지만, 이런 경우라 하더라도 '-하다'와 '-없다'가 여러 어근과 다양하게 결합할 수 있기 때문에 원형을 밝혀 적는 것이 의미를 알기 쉽다. 따라서 '푸카다'가 아닌 '푹하다'로 적는다.\n"
            "[예시3]\n"
            "질문: \"아무 일도 일어나지 않자 그는 {슬몃이/슬며시} 고개를 들어 보았다.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "답변: \"아무 일도 일어나지 않자 그는 슬며시 고개를 들어 보았다.\"가 옳다. '-하다'가 붙는 어근에 '-히'나 '-이'가 붙어서 부사가 되면 어근의 원형을 밝혀 적고, '-하다'가 붙지 않는 어근의 경우라면 소리대로 적는다. '슬몃하다'는 사전에 실려 있지 않은 단어이므로 '슬몃'에 '-이'가 붙어 만들어진 부사는 소리 나는 대로 '슬며시'라고 적는다.\n"
        )
    }
    instruction = type_instructions.get(inp["question_type"], "")
    chat_parts = [instruction, f"[질문]\n{inp['question']}"]
    return "\n\n".join(chat_parts)

def make_chat_cot(inp: dict) -> str:
    """JSON 한 항목(input)에 대해 user 프롬프트 문자열 생성"""
    type_instructions = {
        "선다형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
            "[예시]\n"
            "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
            "1) 주사위 놀이\n"
            "2) 검무\n"
            "3) 격구\n"
            "4) 영고\n"
            "5) 무애무\n"
            "답변: 3"
        ),
        "서술형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
            "[예시]\n"
            "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
            "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
            "[예시]\n"
            "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
            "답변: 정약용"
        ),
        "교정형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 답변하기 전에 '생각 과정'을 작성하고 '답변'을 한국어로 작성하시오.\n"
            "1. 먼저 단계별 생각 과정을 '생각 과정:' 다음에 자유롭게 작성합니다.\n"
            "2. 마지막에 답변을 '답변:' 다음에 \"…\"가 옳다. 이유…\n 형식으로 작성합니다.\n"
            "[예시1]\n"
            "[질문] 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
            "생각 과정: (예시에서는 생략)\n"
            "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다.\n"
            "[예시2]\n"
            "[질문] 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"공부를 하던지 책을 읽던지 하고 싶은 걸 해라.\"\n"
            "생각 과정: (예시에서는 생략)\n"
            "답변: \"공부를 하든지 말든지 네 마음대로 해라.\"가 옳다. '-던'은 과거의 일을 전달할 때 사용하고 '-든'은 선택이나 무관의 뜻을 나타낼 때 사용한다. 마찬가지로 '-든'이 포함된 '-든지', '-든가' 등은 선택이나 무관의 뜻을, '-던'이 포함된 '-던지', '-던데' 등은 과거를 나타낸다. 이 문장은 공부를 하는 것과 마는 것 중 선택하라는 맥락을 담고 있으므로 '-든'을 써야 한다.\n"
            "[예시3]\n"
            "[질문] 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"헷갈리게시리 굴지 마라.\"\n"
            "생각 과정: (예시에서는 생략)\n"
            "답변: \"헷갈리게끔 굴지 마라.\"가 옳다. 같은 의미로 사용되는 '-게끔'과 '-게시리' 중 '-게시리'는 꽤 많이 쓰이는 편이나 표준어에서 제외되었다. 방언형이기도 할뿐더러, 같은 의미의 어미 '-도록'이 널리 쓰이고 있어 '-게끔' 하나만 표준어로 삼아도 충분하다고 판단하였기 때문이다."
        ),
        "선택형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 답변하기 전에 '생각 과정'을 작성하고 '답변'을 한국어로 작성하시오.\n"
            "1. 먼저 단계별 생각 과정을 '생각 과정:' 다음에 자유롭게 작성합니다.\n"
            "2. 마지막에 답변을 '답변:' 다음에 \"…\"가 옳다. 이유…\n 형식으로 작성합니다.\n"
            "[예시1]\n"
            "[질문] \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "생각 과정: (예시에서는 생략)\n"
            "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다.\n"
            "[예시2]\n"
            "[질문] \"오늘은 날씨가 {푹하다/푸카다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "생각 과정: (예시에서는 생략)\n"
            "답변: \"오늘은 날씨가 푹하다.\"가 옳다. '-하다'나 '-없다'가 붙어서 된 용언은 그 '-하다'나 '-없다'를 밝히어 적는다. '푹하다'의 '푹'은 '노래하다'의 '노래'와 달리 자립적이지 않은 어근이지만, 이런 경우라 하더라도 '-하다'와 '-없다'가 여러 어근과 다양하게 결합할 수 있기 때문에 원형을 밝혀 적는 것이 의미를 알기 쉽다. 따라서 '푸카다'가 아닌 '푹하다'로 적는다.\n"
            "[예시3]\n"
            "[질문] \"아무 일도 일어나지 않자 그는 {슬몃이/슬며시} 고개를 들어 보았다.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "생각 과정: (예시에서는 생략)\n"
            "답변: \"아무 일도 일어나지 않자 그는 슬며시 고개를 들어 보았다.\"가 옳다. '-하다'가 붙는 어근에 '-히'나 '-이'가 붙어서 부사가 되면 어근의 원형을 밝혀 적고, '-하다'가 붙지 않는 어근의 경우라면 소리대로 적는다. '슬몃하다'는 사전에 실려 있지 않은 단어이므로 '슬몃'에 '-이'가 붙어 만들어진 부사는 소리 나는 대로 '슬며시'라고 적는다.\n"
        )
    }
    instruction = type_instructions.get(inp["question_type"], "")
    chat_parts = [instruction, f"[질문]\n{inp['question']}"]
    return "\n\n".join(chat_parts)

# ---------------------------------------------------------------------
# 2. 데이터셋 로딩 ------------------------------------------------------
def load_json_for_grpo(path: str, use_cot: bool) -> Dataset:
    """
    JSON 파일(path)을 읽어 GRPOTrainer가 요구하는
    {"prompt": messages, "reference_answer": str} 항목으로 변환
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for ex in raw:
        if use_cot:
            user_prompt = make_chat_cot(ex["input"])
        else:
            user_prompt = make_chat(ex["input"])
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user",    "content": user_prompt},
        ]

        rows.append(
            {
                "prompt": messages,                    # 대화 형식
                "reference_answer": ex["output"]["answer"],
            }
        )

    return Dataset.from_list(rows)

# ---------------------------------------------------------------------
# 3. 보상 함수 정의 -----------------------------------------------------
_COT_SPLIT_PATS = [
    r"\[답변\]",                # ← 새 태그
    r"###\s*정답\s*[:：]",      # (이전 태그가 남아 있다면)
    r"\[답변\]\s*[:：]",
    r"\n답변\s*[:：]",
    r"정답\s*[:：]",
    r"답변\s*[:：]",
    r"^답:\s*",
]
_ANSWER_SPLIT_PATS = [r"가 옳다", r"이 옳다"]
_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

def _remove_cot(text: str) -> str:
    # 문자열이 아니면 바로 변환 시도
    if not isinstance(text, str):
        text = _completion_to_text(text)

    last = -1
    for pat in _COT_SPLIT_PATS:
        for m in re.finditer(pat, text, flags=re.I | re.M):
            last = max(last, m.end())
    return text[last:].strip() if last >= 0 else text.strip()

def _extract_answer_reason(text:str):
    ans_pat = r"###\s*답변\s*[:：]"
    why_pat = r"###\s*이유\s*[:：]"
    if re.search(ans_pat, text):
        after_ans = re.split(ans_pat, text, 1)[1].strip()
        if re.search(why_pat, after_ans):
            ans, reason = re.split(why_pat, after_ans, 1)
            return ans.strip(), reason.strip()
    for pat in _ANSWER_SPLIT_PATS:
        idx = text.find(pat)
        if idx != -1:
            split_pt = idx + len(pat)
            ans = text[:split_pt].strip()
            reason = text[split_pt:].lstrip(" ,.:").strip()
            return ans, reason
    return text.strip(), ""

def _rouge1_f1(ref: str, hyp: str) -> float:
    if not ref or not hyp:
        return 0.0
    return _rouge.score(ref, hyp)["rouge1"].fmeasure

def _completion_to_text(pred):
    """
    GRPO가 반환하는 generation 객체를 사람이 읽을 수 있는 문자열로 바꾼다.
    pred – ① str ② [{"role": "...", "content": "..."} , ...]
    """
    if isinstance(pred, str):
        return pred
    if isinstance(pred, dict):
        return pred['content']
    if isinstance(pred, list):
        return " ".join(m.get("content", "") for m in pred)
    # 그 외엔 빈 문자열 반환
    return ""

# 3-2. 실제 보상 함수
def korean_cot_reward(prompts, completions, reference_answer,
                      em_weight=0.5, rouge_weight=0.5, use_cot=False, **_):
    rewards, em_list, rouge_list = [], [], []
    for gt, pred in zip(reference_answer, completions):
        # 리스트 → 문자열
        pred_txt = _completion_to_text(pred)

        if use_cot:
            gt = _remove_cot(gt)
            pred_txt = _remove_cot(pred_txt)

        gt_ans, gt_reason = _extract_answer_reason(gt)
        pr_ans, pr_reason = _extract_answer_reason(pred_txt)
        # print("### gt_ans:", gt_ans, "\ngt_reason:", gt_reason)
        # print("\n### pr_ans:", pr_ans, "\npr_reason:", pr_reason)

        em    = 1.0 if pr_ans.strip() == gt_ans.strip() else 0.0
        rouge = _rouge1_f1(gt_reason, pr_reason)

        em_list.append(em)
        rouge_list.append(rouge)
        rewards.append(em_weight * em + rouge_weight * rouge)
    
    if wandb.run is not None:
        wandb.log({
            "EM":       sum(em_list)/len(em_list),
            "ROUGE1":   sum(rouge_list)/len(rouge_list),
        })

    return rewards

def main(args):
    train_ds = load_json_for_grpo(args.input, args.use_cot)      # 경로 수정
    eval_ds  = None                                       # 필요 없으면 None
    if args.eval_input is not None:
        eval_ds = load_json_for_grpo(args.eval_input, args.use_cot)
    # ---------------------------------------------------------------------
    # 4. GRPO 설정 ---------------------------------------------------------
    model_init_kwargs = {"cache_dir": args.cache_dir,
                         "device_map": "auto",
                         "attn_implementation": "flash_attention_2"
                         }

    bnb_cfg = None
    if args.quant_bits == 8:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant_bits == 4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype = {"fp16":torch.float16,
                                    "bf16":torch.bfloat16,
                                    "fp32":torch.float32}[args.bnb_compute_dtype],
            bnb_4bit_quant_type    = args.bnb_quant_type,      # "nf4" | "fp4"
            bnb_4bit_use_double_quant = args.bnb_use_double_quant,
            llm_int8_skip_modules=["lm_head"]
        )
    
    if bnb_cfg:
        model_init_kwargs["quantization_config"] = bnb_cfg

    cfg = GRPOConfig(
        output_dir            = args.output,
        num_train_epochs      = args.epochs,
        learning_rate         = args.learning_rate,
        lr_scheduler_type     = args.lr_scheduler_type,
        warmup_steps          = args.warmup_steps,
        warmup_ratio          = args.warmup_ratio,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.num_generations,
        steps_per_generation  = args.steps_per_gen,
        num_generations       = args.num_generations,
        max_prompt_length     = args.max_prompt_len,
        max_completion_length = args.max_completion_len,
        temperature           = args.temperature,
        top_p                 = args.top_p,
        gradient_accumulation_steps = args.grad_accum,
        beta                  = args.beta,
        epsilon               = args.eps_low,
        epsilon_high          = args.eps_high,
        delta                 = args.delta_cap,
        num_iterations        = args.num_iter,
        loss_type             = args.loss_type,
        logging_steps         = args.logging_steps,
        save_steps            = args.save_steps,
        eval_strategy   = "epoch",
        seed                  = args.seed,
        report_to             = args.report_to,
        gradient_checkpointing = True if args.gradient_checkpointing else False,
        gradient_checkpointing_kwargs = {"use_reentrant": False},  # 선택
        model_init_kwargs = model_init_kwargs,
        disable_tqdm = False
    )
    cfg.use_liger_loss = args.use_liger

    reward_funcs = functools.partial(
        korean_cot_reward,
        em_weight=args.em_weight,
        rouge_weight=args.rouge_weight,
        use_cot=args.use_cot
    )
    reward_funcs.__name__ = "korean_cot_reward"

    # ---------------------------------------------------------------------
    # 6. Trainer 생성 & 학습 ----------------------------------------------
    peft_cfg = None
    if args.use_lora:
        peft_cfg = LoraConfig(
            task_type       = TaskType.CAUSAL_LM,
            r               = args.lora_r,
            lora_alpha      = args.lora_alpha,
            lora_dropout    = args.lora_dropout,
            target_modules  = args.lora_target_modules,
            bias            = "none",          # LoRA 가중치만 학습
        )

    if "wandb" in args.report_to:
        wandb.init(project=args.wandb_project_name,
                   name=args.wandb_run_name,
                   config=vars(args),           # CLI 전체를 config 로 저장
                   resume="allow")

    trainer = CustomGRPOTrainer(
        model          = args.model_id,
        reward_funcs   = reward_funcs,
        train_dataset  = train_ds,
        eval_dataset   = eval_ds,
        args           = cfg,
        entropy_coef   = args.entropy_coef,
        ce_coef        = args.ce_coef,
        peft_config    = peft_cfg,
        em_weight     = args.em_weight,
        rouge_weight  = args.rouge_weight,
        use_cot       = args.use_cot,
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    exit(main(args))