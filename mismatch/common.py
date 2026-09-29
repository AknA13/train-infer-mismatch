"""Prompts, datasets and answer checking (adapted from adaptive-specdec/data/common.py).

The mismatch corpus is sampled from gsm8k TRAIN so the same prompts can feed
GRPO later; gsm8k TEST and MATH-500 stay eval-only for the RL stage.
"""
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

MATH_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def text_hash(s):
    return hashlib.sha256(norm_text(s).encode()).hexdigest()


def _gsm8k_answer(raw):
    s = str(raw)
    return s.split("####")[-1].strip().replace(",", "") if "####" in s else s.strip()


def load_problems(name, split, n=0, seed=0):
    """name in {'gsm8k','math500'}. Returns [{idx, problem, answer, source}].
    n>0 takes a seeded random subset so 512 problems are not the first 512."""
    from datasets import load_dataset
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        recs = [(r["question"], _gsm8k_answer(r["answer"])) for r in ds]
    elif name == "math500":
        ds = load_dataset(C.MATH500_ID, split=split)
        recs = [(r["problem"], str(r["answer"])) for r in ds]
    else:
        raise ValueError(f"unknown dataset {name!r} (cached here: gsm8k, math500)")
    out = [{"idx": i, "problem": p, "answer": a, "source": f"{name}/{split}"}
           for i, (p, a) in enumerate(recs)]
    if n and n < len(out):
        import random
        rng = random.Random(seed)
        out = rng.sample(out, n)
    return out


def build_prompt(tokenizer, problem, instruction=MATH_INSTRUCTION, thinking=True):
    msgs = [{"role": "user", "content": problem.strip() + instruction}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_boxed(text):
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def answer_after_think(text):
    k = text.rfind(C.THINK_CLOSE)
    return text[k + len(C.THINK_CLOSE):] if k != -1 else text


def _norm_ans(s):
    s = (s or "").strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\,", ""), ("\\!", ""),
                 ("\\ ", ""), ("$", ""), ("%", ""), (",", "")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s).rstrip(".")


def verify_answer(pred_text, gold):
    """True if the prediction matches gold. math-verify first, string fallback."""
    pred = extract_boxed(pred_text)
    if pred is None:
        pred = answer_after_think(pred_text).strip()
    gold_str = str(gold)
    gb = extract_boxed(gold_str)
    if gb is not None:
        gold_str = gb
    try:
        from math_verify import parse, verify
        if verify(parse("\\boxed{" + gold_str + "}"), parse("\\boxed{" + (pred or "") + "}")):
            return True
    except Exception:
        pass
    return pred is not None and _norm_ans(pred) == _norm_ans(gold_str)


# ---- corpus I/O --------------------------------------------------------------
def corpus_paths(model_id, variant=""):
    """(dir, samples.jsonl, meta.json). variant='bi' -> samples_bi.jsonl / meta_bi.json."""
    d = C.CORPUS_DIR / C.model_tag(model_id)
    suf = f"_{variant}" if variant else ""
    return d, d / f"samples{suf}.jsonl", d / f"meta{suf}.json"


def read_corpus(path):
    import json
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
