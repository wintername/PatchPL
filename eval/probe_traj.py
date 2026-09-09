#!/usr/bin/env python3
"""Probe: does base-DS model emit training-format tool calls? (ASCII only)

Feed training-style prompt (OpenCode system + issue), generate, inspect
whether output contains well-formed <tool_call> JSON targeting the right files.
"""
import json, os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "checkpoints/coder-3b-base-ds"
TEST_DATA = "data/swe_verify/swe_verify_test.json"
MAX_NEW_TOKENS = 4096
PROBE_IDS = ["django__django-16333", "astropy__astropy-12907",
             "django__django-12419"]


def load_system():
    import gzip, glob
    files = sorted(glob.glob("data/nemotron/train-*.jsonl.gz"))
    with gzip.open(files[0], "rt", encoding="utf-8") as f:
        r = json.loads(f.readline())
        return str(r["messages"][0]["content"])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    sys_text = load_system()
    with open(TEST_DATA) as f:
        test = {x["instance_id"]: x for x in json.load(f)}

    for iid in PROBE_IDS:
        item = test.get(iid)
        if not item:
            print("MISSING", iid)
            continue
        ps = item["problem_statement"]
        prompt = ("<|im_start|>system\n" + sys_text +
                  "<|im_end|>\n<|im_start|>user\n" + ps +
                  "<|im_end|>\n<|im_start|>assistant\n")
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=0.2, top_p=0.95, do_sample=True,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=False)
        calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>",
                           gen, flags=re.DOTALL)
        names, good_json, files = [], 0, []
        for c in calls:
            body = c.strip().split("\n", 1)
            nm = body[0].strip() if body else "?"
            args_txt = body[1].strip() if len(body) > 1 else ""
            names.append(nm)
            try:
                a = json.loads(args_txt)
                good_json += 1
                if "filePath" in a:
                    files.append(a["filePath"])
            except Exception:
                pass
        print("\n=== %s ===" % iid)
        print("gen chars:", len(gen), "| tool_calls:", len(calls),
              "| valid JSON:", good_json, "| names:", names[:8])
        print("files:", files[:6])
        print("head:", gen[:400].replace("\n", " ")[:400])


if __name__ == "__main__":
    main()
