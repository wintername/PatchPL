#!/usr/bin/env python3
"""Eval base-DS model with the SAME prompt format used in SFT (ASCII only).

1. prompt = training-identical OpenCode system + uploaded-repo wrapper + issue
2. generate full trajectory (greedy)
3. extract patch: direct diff text, or apply write/edit tool calls on
   original files fetched from GitHub raw at base_commit
4. save predictions for the official swebench harness
"""
import difflib
import gzip
import glob
import json
import os
import re
import urllib.request

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "checkpoints/coder-3b-base-ds"
TEST_DATA = "data/swe_verify/swe_verify_test.json"
OUT = "predictions/coder-3b-base-ds-traj.json"
MAX_NEW_TOKENS = 6144
BATCH = 4


def load_prompt_templates():
    files = sorted(glob.glob("data/nemotron/train-*.jsonl.gz"))
    with gzip.open(files[0], "rt", encoding="utf-8") as f:
        r = json.loads(f.readline())
    sys_text = str(r["messages"][0]["content"])
    user_text = str(r["messages"][1]["content"])
    i1 = user_text.find("<issue>")
    i2 = user_text.find("</issue>")
    pre = user_text[:i1 + len("<issue>")]
    post = user_text[i2:]
    return sys_text, pre, post


def fetch_raw(repo, commit, path):
    url = "https://raw.githubusercontent.com/%s/%s/%s" % (repo, commit, path)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def apply_calls(calls, repo, commit):
    """calls: list of {'filePath','content'} or edit dicts. Returns patch str."""
    per_file = {}
    for c in calls:
        p = c.get("filePath", "")
        p = re.sub(r"^/workspace/repo/", "", p)
        if not p:
            continue
        per_file.setdefault(p, []).append(c)
    diffs = []
    for p, cs in per_file.items():
        try:
            orig = fetch_raw(repo, commit, p)
        except Exception:
            continue
        new = orig
        for c in cs:
            if "content" in c and c.get("content") is not None:
                new = str(c["content"])
            else:
                old = str(c.get("oldString", ""))
                nw = str(c.get("newString", ""))
                if old and old in new:
                    new = new.replace(old, nw, 1)
                else:
                    new = new + nw if not old else new
        if new != orig:
            d = list(difflib.unified_diff(
                orig.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile="a/" + p, tofile="b/" + p))
            if d:
                diffs.append("diff --git a/%s b/%s\n" % (p, p) + "".join(d))
    return "\n".join(diffs)


def extract(generation):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>",
                         generation, flags=re.DOTALL):
        body = m.group(1).strip().split("\n", 1)
        name = body[0].strip()
        args_txt = body[1].strip() if len(body) > 1 else ""
        try:
            args = json.loads(args_txt)
        except Exception:
            continue
        if name in ("write", "edit") and isinstance(args, dict):
            calls.append(args)
    direct = None
    for pat in [r"(diff --git.*?)(?=<\|im_end\|>|$)",
                r"```diff\n(.*?)```", r"```patch\n(.*?)```"]:
        mm = re.search(pat, generation, flags=re.DOTALL)
        if mm:
            direct = mm.group(1).strip()
            break
    return calls, direct


def main():
    sys_text, pre, post = load_prompt_templates()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    test = json.load(open(TEST_DATA))
    print("instances:", len(test))

    preds = []
    n_tool, n_direct, n_editpatch = 0, 0, 0
    for i in tqdm(range(0, len(test), BATCH), desc="gen-traj"):
        batch = test[i:i + BATCH]
        prompts = []
        for x in batch:
            user = pre + "\n" + x["problem_statement"] + "\n" + post
            prompts.append("<|im_start|>system\n" + sys_text +
                           "<|im_end|>\n<|im_start|>user\n" + user +
                           "<|im_end|>\n<|im_start|>assistant\n")
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=6144).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        for j, item in enumerate(batch):
            gen = tok.decode(out[j][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=False)
            calls, direct = extract(gen)
            patch = ""
            if direct:
                patch = direct
                n_direct += 1
            elif calls:
                patch = apply_calls(calls, item["repo"], item["base_commit"])
                if patch:
                    n_editpatch += 1
            if calls:
                n_tool += 1
            preds.append({"instance_id": item["instance_id"],
                          "model_name_or_path": "coder-3b-base-ds-traj",
                          "model_patch": patch})

    with open(OUT, "w") as f:
        json.dump(preds, f, indent=2)
    print("predictions:", len(preds))
    print("with tool calls:", n_tool, "| direct diff:", n_direct,
          "| patch from edits:", n_editpatch,
          "| nonempty patches:",
          sum(1 for p in preds if p["model_patch"].strip()))


if __name__ == "__main__":
    main()
