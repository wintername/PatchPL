# ========== 6. ⭐ v9 核心: 奖励函数 ==========

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
    # ↑ 因子1: 检测\boxed{...}存在
    
    if '<<' in response:
        score += 0.3
    # ↑ 因子2: 检测<<...=...>>计算步骤
    #   GSM8K数据自带这种格式,模型学会了
    
    stripped = response.rstrip()
    if re.search(r'\\boxed\{[^}]*\}\s*$', stripped):
        score += 0.2
    # ↑ 因子3a: 以\boxed{...}结尾→完美收尾
    elif re.search(r'[.!?。！？]\s*$', stripped):
        score += 0.1
    # ↑ 因子3b: 以标点结尾→基本干净
    
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
    # ↑ 从回复中提取答案
    
    gt_n = normalize(ground_truth)
    pred_n = normalize(pred)
    # ↑ 标准化两者以便比较
    
    is_correct = (
        pred is not None and
        pred_n is not None and
        pred_n == gt_n
    )
    # ↑ 判断是否正确: 答案非空且标准化后相等
    #   如: pred="18", gt="18" → is_correct=True
    
    fmt = simple_fmt_score(response)
    # ↑ 计算格式分
    
    if is_correct:
        total = 0.7 + 0.3 * fmt
        # ↑ 正确: 保底0.7 + 格式加分
        #   fmt=0.0 → total=0.70
        #   fmt=0.5 → total=0.85
        #   fmt=1.0 → total=1.00
        return total, 1.0, fmt
    else:
        total = 0.3 * fmt
        # ↑ 错误: 最多0.3(格式再好也没用)
        #   fmt=0.0 → total=0.00
        #   fmt=0.3 → total=0.09
        #   fmt=1.0 → total=0.30
        return total, 0.0, fmt


