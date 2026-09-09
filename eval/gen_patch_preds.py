#!/usr/bin/env python3
"""Generate official-harness predictions for patch models (ASCII only).

Usage: gen_patch_preds.py MODEL_PATH TAG
Greedy decoding, SAME system prompt as SFT.
"""
import json
import os
import re
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = sys.argv[1]
TAG = sys.argv[2]
TEST_DATA = "data/swe_verify/swe_verify_test.json"
OUT = "predictions/%s.json" % TAG
SYS = ("You are an expert software engineer. Given a GitHub issue "
       "description, generate a patch (unified diff) that fixes the "
       "issue. Output only the diff.")


def extract_diff(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    for pat in [r'(diff --git.*?)(?=<\|im_end\|>|$)',
                r'(--- a/.*?)(?=<\|im_end\|>|$)',
                r'```diff\n(.*?)```', r'```patch\n(.*?)```']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return re.sub(r'<\|im_end\|>.*', '', text, flags=re.DOTALL).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    test = json.load(open(TEST_DATA))
    preds = []
    for i in tqdm(range(0, len(test), 8), desc=TAG):
        batch = test[i:i + 8]
        prompts = [
            "<|im_start|>system\n" + SYS + "<|im_end|>\n"
            "<|im_start|>user\n" + x["problem_statement"] +
            "<|im_end|>\n<|im_start|>assistant\n"
            for x in batch
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        for j, item in enumerate(batch):
            gen = tok.decode(out[j][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=False)
            asst = gen.split("<|im_start|>assistant\n")[-1]
            asst = asst.split("<|im_end|>")[0]
            preds.append({"instance_id": item["instance_id"],
                          "model_name_or_path": TAG,
                          "model_patch": extract_diff(asst)})
    os.makedirs("predictions", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(preds, f, indent=2)
    print("DONE", TAG, "nonempty:",
          sum(1 for p in preds if p["model_patch"].strip()), "/", len(preds))


if __name__ == "__main__":
    main()
