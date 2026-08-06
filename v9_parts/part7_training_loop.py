for iteration in range(NUM_ITERATIONS):
    print(f"\n{'='*60}")
    print(f"   ITERATION {iteration+1}/{NUM_ITERATIONS}")
    print(f"{'='*60}")
    data = prompt_splits[iteration]
    prompts = [d[0] for d in data]    # 题目列表
    answers = [d[1] for d in data]    # GT答案列表
    print(f"\n   Step 1: Generate K={K} responses each...")
    all_responses = generate_responses(policy, prompts, k=K)
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
        scores = [combined_reward(r, gt, prompt)[0] for r in responses]
        if max(scores) - min(scores) < 0.01:
            stats['skipped'] += 1
            continue
        best_idx = max(range(K), key=lambda i: scores[i])
        worst_idx = min(range(K), key=lambda i: scores[i])
        if best_idx == worst_idx:
            stats['skipped'] += 1
            continue
        winner = responses[best_idx]
        loser  = responses[worst_idx]
        w_ans = extract_answer_math(winner)
        l_ans = extract_answer_math(loser)
        if w_ans and l_ans and normalize(w_ans) == normalize(l_ans):
            continue
        gap = scores[best_idx] - scores[worst_idx]
        score_range = max(scores) - min(scores)
        raw_wr = 0.5 + 0.5 * min(gap / max(score_range, 0.01), 1.0)
        sim = semantic_similarity([winner], [loser]).item()
        cal_wr = calibrate_win_rate(raw_wr, sim)
        training_pairs.append({
            'prompt': prompt,
            'winner': winner,
            'loser': loser,
            'cal_win_rate': cal_wr,
            'winner_score': scores[best_idx],
            'loser_score': scores[worst_idx],
        })
        stats['sim'].append(sim)
        stats['cal_wr'].append(cal_wr)
        stats['raw_wr'].append(raw_wr)
        stats['winner_r'].append(scores[best_idx])
        stats['loser_r'].append(scores[worst_idx])
        stats['total_comps'] += 1
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
        print("   WARNING: No valid training pairs! Skipping iteration.")
        continue
    print(f"\n   Step 3: S-SPPO training...")
    policy.train()  # 训练模式(开启Dropout等)
    for epoch in range(EPOCHS_PER_ITER):
        random.shuffle(training_pairs)
        acc = {'loss': 0, 'sppo': 0, 'rep': 0}
        for idx, pair in enumerate(training_pairs):
            loss_t, sppo_t, rep_t = ssppo_loss_fn(
                policy, ref_model, tokenizer,
                pair['winner'], pair['loser'],
                pair['cal_win_rate']
            )
            loss_t = loss_t / GRAD_ACCUM
            loss_t.backward()
            acc['loss'] += loss_t.item()
            acc['sppo'] += sppo_t.item()
            acc['rep']  += rep_t.item()
            if (idx + 1) % GRAD_ACCUM == 0 or idx == len(training_pairs) - 1:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 30 == 0:
                    all_metrics.append({
                        'iter': iteration,
                        'step': global_step,
                        'loss': acc['loss'],
                        'sppo': acc['sppo'],
                        'rep': acc['rep']
                    })
                    acc = {'loss': 0, 'sppo': 0, 'rep': 0}
    print("   Updating reference (EMA)...")
    with torch.no_grad():
        ref_params = dict(ref_model.named_parameters())
        for name, pp in policy.named_parameters():
            if name in ref_params and pp.requires_grad:
                rp = ref_params[name]
                rp.data = 0.995 * rp.data + 0.005 * pp.data.to(rp.device)
    ckpt = f"{OUTPUT_DIR}/iter{iteration+1}"
    os.makedirs(ckpt, exist_ok=True)
    policy.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f"   Saved: {ckpt}")
    gc.collect()
    torch.cuda.empty_cache()
print(f"\n[4/4] Saving → {OUTPUT_DIR}")
policy.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n{'='*60}")
print(f"  Done! {OUTPUT_DIR}")
print(f"  Total steps: {global_step}")
print(f"{'='*60}")
