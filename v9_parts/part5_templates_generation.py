# ========== 9. 多视角Prompt模板 ==========

PROMPT_TEMPLATES = [
    # 模板1: 分步推理(最常见策略)
    "Let's solve this step by step, breaking the problem into smaller parts.\n\n"
    "Question: {prompt}\n\nSolution:",
    
    # 模板2: 先识别已知/未知(结构化思维)
    "First, identify the known quantities and the unknown we need to find.\n\n"
    "Question: {prompt}\n\nSolution:",
    
    # 模板3: 反向推理(从目标倒推)
    "Let's work backwards from the goal to find what we need.\n\n"
    "Question: {prompt}\n\nSolution:",
    
    # 模板4: 方程组法(形式化建模)
    "Let's set up equations to represent the problem, then solve them.\n\n"
    "Question: {prompt}\n\nSolution:",
    
    # 模板5: 逻辑推理(每步自检)
    "Let's reason through this logically, checking each step as we go.\n\n"
    "Question: {prompt}\n\nSolution:",
    
    # 模板6: 先估算再精确(双层验证)
    "Let's approach this by first estimating the answer, then calculating precisely.\n\n"
    "Question: {prompt}\n\nSolution:",
]
# ↑ 6个模板对应6种不同的推理策略
#   思路来自DiVeRSe论文: 多角度prompt→增加生成多样性
#   每题用全部6个模板各生成1次→6个回复


# ========== 10. 生成函数 ==========

@torch.no_grad()
def generate_responses(model, prompts, k=K):
    """
    对每个题目,用k个不同模板生成k个回复
    
    参数:
      model:  当前策略模型(Policy + LoRA)
      prompts: 题目列表
      k:      使用的模板数(默认K=6)
    
    返回:
      all_responses[i][j]: 第i题的第j个回复
    """
    all_responses = []
    model.eval()  # 评估模式(关闭Dropout)
    templates = PROMPT_TEMPLATES[:k]  # 取前k个模板
    
    for prompt in tqdm(prompts, desc="   Generating"):
        responses = []
        for tpl in templates:
            text = tpl.format(prompt=prompt)
            # ↑ 将题目填入模板: "Question: {prompt}\n\nSolution:"
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            # ↑ 分词并送到GPU
            
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,   # 最多生成256个新token
                temperature=GEN_TEMPERATURE,      # T=1.0: 标准softmax
                top_p=1.0,                        # nucleus sampling(1.0=不限制)
                do_sample=True,                   # 采样模式(非贪心)
                pad_token_id=tokenizer.pad_token_id,
            )
            # ↑ 自回归生成: 逐个token预测,直到EOS或max_new_tokens
            
            resp = tokenizer.decode(
                out[0][inputs['input_ids'].shape[1]:],
                # ↑ 只解码新生成的部分(去掉输入的prompt)
                skip_special_tokens=True
                # ↑ 跳过<|endoftext|>等特殊token
            )
            responses.append(resp)
        
        all_responses.append(responses)
        # ↑ 第i题: [回复1, 回复2, ..., 回复6]
    
    return all_responses


