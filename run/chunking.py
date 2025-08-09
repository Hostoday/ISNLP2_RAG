import re, json, os
from docx import Document

IN_PATH  = "../data/국어 지식 기반 생성(RAG) 참조 문서.docx"
OUT_PATH = "../data/korean_rule_chunks_final4.json"

# ────────── 패턴 ──────────
TITLE_TAG   = re.compile(r"<.*>")
TOP_HDR     = re.compile(r"^(?:\d+\.\s*|다만\s*\d*|\[붙임\s*\d*\])")
SUB_HDR     = re.compile(r"^\(\d+\)\s*")
BULLET_HDR  = re.compile(r"^\s*￭\s*")

CLEAN_PREFIX = re.compile(r"^\s*(?:￭|\[붙임\s*\d*\]|\[붙임\])\s*")
def clean(line: str) -> str:
    return CLEAN_PREFIX.sub("", line).strip()

def export_chunk(chunks, cid, title, text_lines, examples, original):
    if not title or not text_lines:
        return cid

    text_block = "\n".join(text_lines).strip()
    ex_block   = examples[:]

    # ── ① 문장 부호 규정: 하이픈 뒤 구절을 text 맨 앞에 추가 ──
    m = re.match(r"문장 부호\s*-\s*(.+)", title)
    if m:
        suffix = m.group(1).strip()
        if suffix and not text_block.startswith(suffix):
            text_block = f"{suffix}\n{text_block}" if text_block else suffix
    # ────────────────────────────────────────────────

    # ── ② 외래어 표기법 특수 처리 (기존 그대로) ──
    if "외래어 표기법" in title and "적용한다:" in text_block:
        head, tail = text_block.split("적용한다:", 1)
        text_block = head.strip() + " 적용한다:"
        ex_block  += [ln.strip() for ln in tail.strip().splitlines() if ln.strip()]
    # ────────────────────────────────────────────────

    chunks.append(
        {
            "chunk_id": f"rule-{cid}",
            "rule_title": title,
            "text": text_block,
            "examples": ex_block,
            "original": original[:],
        }
    )
    return cid + 1

# ────────── 변수 초기화 ──────────
doc = Document(IN_PATH)
chunks, cid = [], 1
current_title = ""

# base 버퍼
base_lines, base_ex, base_ori = [], [], []
base_saved, base_prefix = False, ""

# 현재 Top / Sub 상태
top_hdr_clean = top_hdr_raw = ""
sub_hdr_clean = sub_hdr_raw = ""

# 진행 중 special 버퍼
spec_lines, spec_ex, spec_ori = [], [], []
mode = None            # None / base / special

# '다음 빈칸 …' 2줄 예시
skip_next_ex, skip_prefix = False, ""

# ────────── 문서 순회 ──────────
for para in doc.paragraphs:
    raw = para.text.rstrip()
    if not raw:
        continue

    # (0) 두 줄 예시 처리
    if skip_next_ex:
        spec_ex.append(f"{skip_prefix} {raw}")
        spec_ori.append(f"{skip_prefix} {raw}")
        skip_next_ex = False
        continue
    if raw.startswith("- 다음 빈칸에 알맞은 조사를 쓰시오."):
        skip_prefix = raw[2:].strip()
        skip_next_ex = True
        continue

    # (1) 타이틀
    if TITLE_TAG.fullmatch(raw):
        if mode == "special":
            cid = export_chunk(chunks, cid, current_title,
                               spec_lines, spec_ex, spec_ori)
        if base_lines and not base_saved:
            cid = export_chunk(chunks, cid, current_title,
                               base_lines, base_ex, base_ori)

        current_title = raw.strip("<>")
        base_lines, base_ex, base_ori = [], [], []
        base_saved, base_prefix      = False, ""
        top_hdr_clean = top_hdr_raw  = ""
        sub_hdr_clean = sub_hdr_raw  = ""
        spec_lines, spec_ex, spec_ori = [], [], []
        mode = "base"
        base_ori.append(raw)
        continue

    # (2) 예시 '- '
    if raw.startswith("- "):
        (base_ex if mode == "base" else spec_ex).append(raw[2:].strip())
        (base_ori if mode == "base" else spec_ori).append(raw)
        continue

    # (3) BULLET ￭ …  (가장 먼저 분리)
    if BULLET_HDR.match(raw):
        # 진행 중 청크 저장
        if mode == "special":
            cid = export_chunk(chunks, cid, current_title,
                               spec_lines, spec_ex, spec_ori)
        elif mode == "base" and base_lines and not base_saved:
            cid = export_chunk(chunks, cid, current_title,
                               base_lines, base_ex, base_ori)
            base_saved = True
            base_prefix = "\n".join(base_lines).strip()

        # ￭ 줄 자체를 독립 청크로
        bullet_clean = clean(raw)
        cid = export_chunk(
            chunks, cid, current_title,
            [bullet_clean], [], [raw]
        )

        # 새 버퍼 초기화 (bullet 이후 본문을 받을 준비)
        spec_lines, spec_ex, spec_ori = [], [], []
        mode = "special"
        continue

    # (4) TOP 헤더
    if TOP_HDR.match(raw):
        if mode == "base" and base_lines and not base_saved:
            cid = export_chunk(chunks, cid, current_title,
                               base_lines, base_ex, base_ori)
            base_saved  = True
            base_prefix = "\n".join(base_lines).strip()
        if mode == "special":
            cid = export_chunk(chunks, cid, current_title,
                               spec_lines, spec_ex, spec_ori)

        top_hdr_clean = clean(raw)
        top_hdr_raw   = raw
        sub_hdr_clean = sub_hdr_raw = ""

        spec_lines = [base_prefix, top_hdr_clean] if base_prefix else [top_hdr_clean]
        spec_ex, spec_ori = [], [raw]
        mode = "special"
        continue

    # (5) SUB 헤더 ( (1) … )
    if SUB_HDR.match(raw):
        if mode == "special":
            cid = export_chunk(chunks, cid, current_title,
                               spec_lines, spec_ex, spec_ori)

        sub_hdr_clean = clean(raw)
        sub_hdr_raw   = raw

        spec_lines = []
        if base_prefix:   spec_lines.append(base_prefix)
        if top_hdr_clean: spec_lines.append(top_hdr_clean)
        spec_lines.append(sub_hdr_clean)

        spec_ex, spec_ori = [], [sub_hdr_raw]
        mode = "special"
        continue

    # (6) 일반 본문
    if mode == "base":
        base_lines.append(clean(raw)); base_ori.append(raw)
    else:
        spec_lines.append(clean(raw)); spec_ori.append(raw)

# ────────── flush 잔여 버퍼 ──────────
if mode == "special":
    cid = export_chunk(chunks, cid, current_title,
                       spec_lines, spec_ex, spec_ori)
if base_lines and not base_saved:
    cid = export_chunk(chunks, cid, current_title,
                       base_lines, base_ex, base_ori)

# ────────── 저장 ──────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as fp:
    json.dump(chunks, fp, ensure_ascii=False, indent=2)

print(f"✓ 총 {len(chunks)}개 청크 저장 → {OUT_PATH}")
