#!/usr/bin/env python3
"""Patch-bridge SFT (issue -> unified diff) (ASCII only).

Usage: train_patch_sft.py MODEL_PATH OUTPUT_DIR TAG
"""
import json
import os
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_PATH = None
OUTPUT_DIR = None
TAG = None
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL_PATH, OUTPUT_DIR, TAG = _pos[0], _pos[1], _pos[2]
TRAIN_DATA = "data/patch_sft/train.jsonl"
MAX_SEQ_LENGTH = 4096


def fmt(ex):
    return ex["rendered"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = [json.loads(l) for l in open(TRAIN_DATA)]
    kept = [d for d in data
            if len(tok(d["rendered"])["input_ids"]) <= MAX_SEQ_LENGTH - 64]
    print("samples:", len(data), "fit:", len(kept))
    ds = Dataset.from_list(kept)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    args = SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, warmup_steps=10, lr_scheduler_type="cosine",
        logging_steps=4, save_steps=32, save_total_limit=1,
        bf16=True, optim="sgd", gradient_checkpointing=True,
        report_to="none", remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH, deepspeed="ds_config.json",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, formatting_func=fmt)
    has_ckpt = os.path.isdir(OUTPUT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR))
    trainer.train(resume_from_checkpoint=has_ckpt)
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print("PATCH-SFT-DONE-" + TAG)


if __name__ == "__main__":
    main()
