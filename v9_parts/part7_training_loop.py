for iteration in range(NUM_ITERATIONS):
    # ==================== 外层循环: 3轮迭代 ====================
    
    print(f"\n{'='*60}")
    print(f"   ITERATION {iteration+1}/{NUM_ITERATIONS}")
    print(f"{'='*60}")

    data = prompt_splits[iteration]
    prompts = [d[0] for d in data]    # 题目列表
    answers = [d[1] for d in data]    # GT答案列表

    # ---- Step 1: 生成 ----
    print(f"\n   Step 1: Generate K={K} responses each...")
    all_responses = generate_responses(policy, prompts, k=K)
    # ↑ 返回: [[回复1_1,...,回复1_6], [回复2_1,...], ...]
    #   每个题目6个回复,共100题→600条回复

    # ---- Step 2: 评分 + 选Pair ----
    print(f"\n   Step 2: v9 simple scoring (correctness-gated) + pair selection")

    training_pairs = []
    stats = {
        'sim': [],      # 语义相似度列表
        'cal_wr': [],   # 校准后win rate列表
        'raw_wr': [],   # 原始win rate列表
        'skipped': 0,   # 跳过的题目数
        'total_comps': 0,
        'winner_r': [], # Winner得分列表
        'loser_r': [],  # Loser得分列表
    }

    for prompt, gt, responses in tqdm(
            zip(prompts, answers, all_responses),
            total=len(prompts),
            desc="   Pairs"):
        # ===== 对每道题 =====
        
        # --- 计算6个回复的得分 ---
        scores = [combined_reward(r, gt, prompt)[0] for r in responses]
        # ↑ 调用v9奖励函数: 正确→0.7~1.0, 错误→0.0~0.3
        
        if max(scores) - min(scores) < 0.01:
            # ↑ 所有回复分数太接近(差距<0.01)→无区分度→跳过
            stats['skipped'] += 1
            continue

        # --- 找最好和最差的回复 ---
        best_idx = max(range(K), key=lambda i: scores[i])
        # ↑ 分数最高的回复索引(Winner)
        worst_idx = min(range(K), key=lambda i: scores[i])
        # ↑ 分数最低的回复索引(Loser)
        
        if best_idx == worst_idx:
            # ↑ 最好=最差(所有分数相同)→跳过
            stats['skipped'] += 1
            continue

        winner = responses[best_idx]
        loser  = responses[worst_idx]
        
        # --- 安全检查: 跳过答案相同的pair ---
        w_ans = extract_answer_math(winner)
        l_ans = extract_answer_math(loser)
        if w_ans and l_ans and normalize(w_ans) == normalize(l_ans):
            # ↑ 两个回复答案相同→无对比价值→跳过
            continue

        # --- 计算Win Rate ---
        gap = scores[best_idx] - scores[worst_idx]
        # ↑ Winner和Loser的分数差距
        score_range = max(scores) - min(scores)
        # ↑ 6个回复的分数范围
        
        raw_wr = 0.5 + 0.5 * min(gap / max(score_range, 0.01), 1.0)
        # ↑ 原始win rate: 根据相对差距计算
        #   gap=0    → raw_wr=0.5 (无偏向)
        #   gap=range→ raw_wr=1.0 (完胜)
        #   防止除零: max(score_range, 0.01)

        sim = semantic_similarity([winner], [loser]).item()
        # ↑ 计算winner/loser的文本相似度
        cal_wr = calibrate_win_rate(raw_wr, sim)
        # ↑ 语义校准: 太相似→降低win rate

        # --- 添加到训练pair列表 ---
        training_pairs.append({
            'prompt': prompt,
            'winner': winner,
            'loser': loser,
            'cal_win_rate': cal_wr,
            'winner_score': scores[best_idx],
            'loser_score': scores[worst_idx],
        })
        
        # --- 记录统计 ---
        stats['sim'].append(sim)
        stats['cal_wr'].append(cal_wr)
        stats['raw_wr'].append(raw_wr)
        stats['winner_r'].append(scores[best_idx])
        stats['loser_r'].append(scores[worst_idx])
        stats['total_comps'] += 1

    # ---- 打印Step 2统计 ----
    valid = len(prompts) - stats['skipped']
    print(f"   Valid prompts: {valid}, Skipped: {stats['skipped']}")
    print(f"   Training pairs: {len(training_pairs)}")
    print(f"   Avg similarity: {np.mean(stats['sim']):.4f}" if stats['sim'] else "   No pairs!")
    if stats['sim']:
        print(f"   Raw WR {np.mean(stats['raw_wr']):.3f} → Calibrated WR {np.mean(stats['cal_wr']):.3f}")
    if stats['winner_r']:
        print(f"   Winner score: {np.mean(stats['winner_r']):.3f}, "
              f"Loser score: {np.mean(stats['loser_r']):.3f}")

    if not training_pairs:
        # ↑ 没有有效pair→跳过本轮训练
        print("   WARNING: No valid training pairs! Skipping iteration.")
        continue

    # ---- Step 3: S-SPPO 训练 ----
    print(f"\n   Step 3: S-SPPO training...")
    policy.train()  # 训练模式(开启Dropout等)

    for epoch in range(EPOCHS_PER_ITER):
        # ===== 内层循环: 2个epoch =====
        random.shuffle(training_pairs)
        # ↑ 打乱训练对顺序(每epoch不同顺序)
        
        acc = {'loss': 0, 'sppo': 0, 'rep': 0}
        # ↑ 累积loss用于日志

        for idx, pair in enumerate(training_pairs):
            # ===== 逐对训练 =====
            
            loss_t, sppo_t, rep_t = ssppo_loss_fn(
                policy, ref_model, tokenizer,
                pair['winner'], pair['loser'],
                pair['cal_win_rate']
            )
            # ↑ 计算SPPO损失: (r_w-Δ)²+(r_l+Δ)²+λ·cos(h_w,h_l)
            
            loss_t = loss_t / GRAD_ACCUM
            # ↑ 梯度累积: loss除以累积步数
            #   等效于batch_size=8的平均loss
            
            loss_t.backward()
            # ↑ 反向传播,累积梯度(不清零)

            acc['loss'] += loss_t.item()
            acc['sppo'] += sppo_t.item()
            acc['rep']  += rep_t.item()

            # --- 梯度累积触发更新 ---
            if (idx + 1) % GRAD_ACCUM == 0 or idx == len(training_pairs) - 1:
                # ↑ 每8步或最后一步→更新参数
                
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                # ↑ 梯度裁剪: 防止梯度爆炸, max_norm=1.0
                
                optimizer.step()
                # ↑ 更新LoRA参数
                
                scheduler.step()
                # ↑ 更新学习率(线性预热+衰减)
                
                optimizer.zero_grad()
                # ↑ 清空梯度,准备下一轮累积
                
                global_step += 1
                # ↑ 全局步数+1

                # --- 每30步记录一次 ---
                if global_step % 30 == 0:
                    all_metrics.append({
                        'iter': iteration,
                        'step': global_step,
                        'loss': acc['loss'],
                        'sppo': acc['sppo'],
                        'rep': acc['rep']
                    })
                    acc = {'loss': 0, 'sppo': 0, 'rep': 0}
                    # ↑ 重置累积loss

    # ---- EMA更新参考模型 ----
    print("   Updating reference (EMA)...")
    with torch.no_grad():
        ref_params = dict(ref_model.named_parameters())
        for name, pp in policy.named_parameters():
            if name in ref_params and pp.requires_grad:
                rp = ref_params[name]
                rp.data = 0.995 * rp.data + 0.005 * pp.data.to(rp.device)
                # ↑ EMA: ref = 0.995×ref + 0.005×policy
                #   平滑更新,防止参考模型突变

    # ---- 保存checkpoint ----
    ckpt = f"{OUTPUT_DIR}/iter{iteration+1}"
    os.makedirs(ckpt, exist_ok=True)
    policy.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f"   Saved: {ckpt}")

    # ---- 释放显存 ----
    gc.collect()
    torch.cuda.empty_cache()


# ========== 14. 最终保存 ==========

print(f"\n[4/4] Saving → {OUTPUT_DIR}")

policy.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# ↑ 保存最终模型(第3轮训练后的权重)

# 保存训练指标
with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
# ↑ 保存每30步的loss/sppo/rep指标

print(f"\n{'='*60}")
print(f"  Done! {OUTPUT_DIR}")
print(f"  Total steps: {global_step}")
print(f"{'='*60}")
# ==================== 代码结束 ====================
