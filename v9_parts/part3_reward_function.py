def simple_fmt_score(response):
    """
    v9 3因子简化格式评分 — 满分1.0
    三个因子:
      1. \boxed{} 存在        → +0.5 (数学答案标准格式)
      2. <<...=...>> 计算步骤  → +0.3 (逐步推理展示)
      3. 干净收尾              → +0.2 (以\boxed结尾或标点结尾)
    ⚠️ 已知问题:
      模型训练时没见过\boxed{}格式(SFT用的是<answer>标签)
      所以因子1几乎永远=0, 因子3也常=0
      导致fmt几乎永远=0.30, 失去区分度
    """
    score = 0.0
    if '\\boxed{' in response:
        score += 0.5
    if '<<' in response:
        score += 0.3
    stripped = response.rstrip()
    if re.search(r'\\boxed\{[^}]*\}\s*$', stripped):
        score += 0.2
    elif re.search(r'[.!?。！？]\s*$', stripped):
        score += 0.1
    return score
def combined_reward(response, ground_truth, prompt=""):
    """
    v9 正确性硬门控奖励 — 核心设计!
    设计原理:
      正确答案无条件 > 错误答案
      正确区间 [0.7, 1.0] 完全高于 错误区间 [0.0, 0.3]
    返回值:
      (total_score, math_score, fmt_score)
      total_score: 最终得分
      math_score:  1.0=正确, 0.0=错误
      fmt_score:   格式分
    """
    pred = extract_answer_math(response)
    gt_n = normalize(ground_truth)
    pred_n = normalize(pred)
    is_correct = (
        pred is not None and
        pred_n is not None and
        pred_n == gt_n
    )
    fmt = simple_fmt_score(response)
    if is_correct:
        total = 0.7 + 0.3 * fmt
        return total, 1.0, fmt
    else:
        total = 0.3 * fmt
        return total, 0.0, fmt
