#!/usr/bin/env python3
"""Build issue->patch SFT data (ASCII only). Same prompt as test time."""
import json

FULL = "data/swe_verify/swe_verify_full.json"
TEST = "data/swe_verify/swe_verify_test.json"
OUT = "data/patch_sft/train.jsonl"
SYS = ("You are an expert software engineer. Given a GitHub issue "
       "description, generate a patch (unified diff) that fixes the "
       "issue. Output only the diff.")


def main():
    import os
    os.makedirs("data/patch_sft", exist_ok=True)
    test_ids = {x["instance_id"] for x in json.load(open(TEST))}
    rows = [x for x in json.load(open(FULL))
            if x["instance_id"] not in test_ids]
    with open(OUT, "w", encoding="utf-8") as f:
        for x in rows:
            rendered = ("<|im_start|>system\n" + SYS + "<|im_end|>\n"
                        "<|im_start|>user\n" + x["problem_statement"] +
                        "<|im_end|>\n<|im_start|>assistant\n" +
                        x["patch"] + "<|im_end|>")
            f.write(json.dumps({"uuid": x["instance_id"],
                                "rendered": rendered},
                               ensure_ascii=False) + "\n")
    print("patch_sft samples:", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
