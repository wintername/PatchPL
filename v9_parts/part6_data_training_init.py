print("\n[2/4] Loading prompts...")
gsm8k = load_dataset("gsm8k", "main", split="train")
all_data = [
    (gsm8k[i]['question'],
     gsm8k[i]['answer'].split("####")[-1].strip())
    for i in range(min(len(gsm8k),
                       PROMPTS_PER_ITER * NUM_ITERATIONS + 500))
]
prompt_splits = []
for i in range(NUM_ITERATIONS):
    s = i * PROMPTS_PER_ITER          # 起始索引: 0, 100, 200
    e = (i + 1) * PROMPTS_PER_ITER    # 结束索引: 100, 200, 300
    prompt_splits.append(all_data[s:e])
    print(f"   Iter {i+1}: {len(prompt_splits[-1])} prompts")
print("\n[3/4] Starting S-SPPO training (v9 simplified)...")
optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
total_steps = (
    NUM_ITERATIONS * PROMPTS_PER_ITER * EPOCHS_PER_ITER
    // (BATCH_SIZE * GRAD_ACCUM)
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 前10%步数线性预热
    num_training_steps=total_steps             # 之后线性衰减到0
)
print(f"   Pair mode: SIMPLE (v9: correctness-gated, no anchor improvement)")
print(f"   Reward: 正确→0.7+0.3*fmt, 错误→0.3*fmt")
print(f"   Temperature: {GEN_TEMPERATURE}")
global_step = 0
all_metrics = []
