# ========== 11. 数据加载 ==========

print("\n[2/4] Loading prompts...")

gsm8k = load_dataset("gsm8k", "main", split="train")
# ↑ 加载GSM8K训练集(7473道小学数学应用题)

all_data = [
    (gsm8k[i]['question'],
     gsm8k[i]['answer'].split("####")[-1].strip())
    # ↑ answer格式: "推理过程 #### 18" → 取"18"
    for i in range(min(len(gsm8k),
                       PROMPTS_PER_ITER * NUM_ITERATIONS + 500))
    # ↑ 取前800题(3轮×100题+500余量)
]

prompt_splits = []
# ↑ 将题目分成3份,每轮用不同的100题
for i in range(NUM_ITERATIONS):
    s = i * PROMPTS_PER_ITER          # 起始索引: 0, 100, 200
    e = (i + 1) * PROMPTS_PER_ITER    # 结束索引: 100, 200, 300
    prompt_splits.append(all_data[s:e])
    print(f"   Iter {i+1}: {len(prompt_splits[-1])} prompts")


# ========== 12. 训练初始化 ==========

print("\n[3/4] Starting S-SPPO training (v9 simplified)...")

optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
# ↑ AdamW优化器: Adam + 权重衰减(解耦)
#   只优化policy.parameters()中requires_grad=True的(LoRA参数)

total_steps = (
    NUM_ITERATIONS * PROMPTS_PER_ITER * EPOCHS_PER_ITER
    // (BATCH_SIZE * GRAD_ACCUM)
)
# ↑ 总训练步数: 3轮 × 100题 × 2epoch ÷ 8accum = 75步

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 前10%步数线性预热
    num_training_steps=total_steps             # 之后线性衰减到0
)
# ↑ 学习率调度: 预热阶段从0→lr, 然后线性衰减→0

print(f"   Pair mode: SIMPLE (v9: correctness-gated, no anchor improvement)")
print(f"   Reward: 正确→0.7+0.3*fmt, 错误→0.3*fmt")
print(f"   Temperature: {GEN_TEMPERATURE}")

global_step = 0
# ↑ 全局训练步数计数器(跨迭代)
all_metrics = []
# ↑ 存储训练指标(loss, sppo_loss, rep_loss)


# ========== 13. ⭐ 训练主循环 ==========

