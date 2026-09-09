#!/usr/bin/env python3
"""SFT Qwen2.5-Coder-3B BASE model on best single PASS trajectories (ASCII).

Trains the base model on chat-markup agent trajectories (best_sft.jsonl),
behavior cloning: reasoning + tool calls + observations, pass-only.
"""
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "models/Qwen2.5-Coder-3B"
TRAIN_DATA = "data/nemotron/deepseek_sft_chunks.jsonl"
OUTPUT_DIR = "checkpoints/coder-3b-base-ds"
MAX_SEQ_LENGTH = 32768


def fmt(ex):
    return ex["rendered"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = [json.loads(l) for l in open(TRAIN_DATA)]
    print("deepseek_sft samples:", len(data))
    kept = []
    for d in data:
        n = len(tok(d["rendered"])["input_ids"])
        if n <= MAX_SEQ_LENGTH - 64:
            kept.append(d)
    print("fit in seq(%d):" % MAX_SEQ_LENGTH, len(kept), "of", len(data))
    ds = Dataset.from_list(kept)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    args = SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, warmup_steps=20, lr_scheduler_type="cosine",
        logging_steps=5, save_steps=60, save_total_limit=2,
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
    print("BASE-SFT-DONE")


if __name__ == "__main__":
    main()
