PROMPT_TEMPLATES = [
    "Let's solve this step by step, breaking the problem into smaller parts.\n\n"
    "Question: {prompt}\n\nSolution:",
    "First, identify the known quantities and the unknown we need to find.\n\n"
    "Question: {prompt}\n\nSolution:",
    "Let's work backwards from the goal to find what we need.\n\n"
    "Question: {prompt}\n\nSolution:",
    "Let's set up equations to represent the problem, then solve them.\n\n"
    "Question: {prompt}\n\nSolution:",
    "Let's reason through this logically, checking each step as we go.\n\n"
    "Question: {prompt}\n\nSolution:",
    "Let's approach this by first estimating the answer, then calculating precisely.\n\n"
    "Question: {prompt}\n\nSolution:",
]
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
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,   # 最多生成256个新token
                temperature=GEN_TEMPERATURE,      # T=1.0: 标准softmax
                top_p=1.0,                        # nucleus sampling(1.0=不限制)
                do_sample=True,                   # 采样模式(非贪心)
                pad_token_id=tokenizer.pad_token_id,
            )
            resp = tokenizer.decode(
                out[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(resp)
        all_responses.append(responses)
    return all_responses
