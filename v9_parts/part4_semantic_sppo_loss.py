def semantic_similarity(texts_a, texts_b):
    """
    Token-overlap文本相似度 — 替代SentenceTransformer
    算法: 70% Jaccard相似度 + 30% 长度惩罚
    Jaccard = |A∩B| / |A∪B|
    例: A={dog,cat,bird}, B={cat,bird,fish}
        Jaccard = 2/4 = 0.5
    为什么需要相似度?
      语义校准时用: 两个回复太相似→偏好信号可能是噪声
      → calibrate_win_rate把win_rate往0.5拉
    """
    if isinstance(texts_a, list):
        return torch.tensor([
            semantic_similarity(a, b) for a, b in zip(texts_a, texts_b)
        ])
    def tokenize(s):
        return set(re.findall(r'\d+|[a-zA-Z]{2,}', s.lower()))
    ta = tokenize(texts_a)
    tb = tokenize(texts_b)
    if not ta or not tb:
        return torch.tensor(0.5)
    jaccard = len(ta & tb) / len(ta | tb)
    len_penalty = min(len(texts_a), len(texts_b)) / max(len(texts_a), len(texts_b), 1)
    sim = 0.7 * jaccard + 0.3 * len_penalty
    return torch.tensor(sim)
def calibrate_win_rate(raw_wr, sim):
    """
    语义校准: 相似度越高→偏好信号越弱
    公式: cal_wr = 0.5 + (1-sim) × (raw_wr - 0.5)
    例:
      raw_wr=1.0, sim=0.0 (完全不同) → cal_wr=1.00 (保持)
      raw_wr=1.0, sim=0.5 (中等相似) → cal_wr=0.75 (减半)
      raw_wr=1.0, sim=0.95(几乎一样) → cal_wr=0.525(几乎归零)
    直觉: 如果winner和loser几乎一样的文本, 
          那"winner更好"的信号很可能是噪声
    """
    sim = max(0.0, min(1.0, sim))
    return 0.5 + (1.0 - sim) * (raw_wr - 0.5)
@torch.no_grad()
def get_hidden_state(model, input_ids, attention_mask):
    """
    获取序列最后一个token的hidden state (用于表征校准)
    用途: 计算 cos(h_winner, h_loser)
    如果两个hidden state太接近→loss_rep项增大→惩罚
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    hidden = outputs.hidden_states[-1]
    last_pos = attention_mask.sum(dim=1) - 1
    h = hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos]
    return F.normalize(h.float(), p=2, dim=-1).to(torch.bfloat16)
def ssppo_loss_fn(policy, ref_model, tokenizer,
                  winner_text, loser_text, cal_win_rate_target,
                  lamb=LAMBDA_REP, bet=BETA):
    """
    S-SPPO核心损失函数
    数学公式:
      L = (r_w - Δ)² + (r_l + Δ)² + λ·cos(h_w, h_l)
    其中:
      r_w = log π_θ(winner|x) - log π_ref(winner|x)  ← 策略模型相对参考模型的增益
      r_l = log π_θ(loser|x)  - log π_ref(loser|x)
      Δ   = (win_rate - 0.5) / β                        ← 目标差距
      cos(h_w, h_l)                                     ← 表征校准项
    直觉: 拉大winner和loser的log概率差距, 同时防止hidden state坍缩
    """
    w_tok = tokenizer(
        winner_text,
        return_tensors="pt",       # 返回PyTorch tensor
        truncation=True,           # 超过max_length则截断
        max_length=512,            # 最大512 tokens
        padding=True               # 自动padding到batch内最长
    ).to(policy.device)           # 送到Policy所在GPU
    l_tok = tokenizer(
        loser_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(policy.device)
    pol_w_out = policy(**w_tok)
    pol_l_out = policy(**l_tok)
    with torch.no_grad():
        w_tok_ref = {k: v.to(ref_model.device) for k, v in w_tok.items()}
        l_tok_ref = {k: v.to(ref_model.device) for k, v in l_tok.items()}
        ref_w_out = ref_model(**w_tok_ref)
        ref_l_out = ref_model(**l_tok_ref)
    def seq_logp(logits, input_ids):
        """
        计算序列的对数概率: Σ log P(token_i | token_{<i})
        原理:
          logits → softmax → log → 取每个位置实际token的概率 → 求和
          只对非padding位置求和
        """
        logp = F.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:]
        lp = logp[:, :-1].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != tokenizer.pad_token_id).float()
        return (lp * mask).sum(dim=-1)
    pol_w_lp = seq_logp(pol_w_out.logits, w_tok['input_ids'])
    pol_l_lp = seq_logp(pol_l_out.logits, l_tok['input_ids'])
    ref_w_lp = seq_logp(ref_w_out.logits, w_tok_ref['input_ids']).to(policy.device)
    ref_l_lp = seq_logp(ref_l_out.logits, l_tok_ref['input_ids']).to(policy.device)
    r_w = pol_w_lp - ref_w_lp
    r_l = pol_l_lp - ref_l_lp
    target = torch.tensor(
        (cal_win_rate_target - 0.5) / bet,
        device=policy.device,
        dtype=torch.bfloat16
    )
    loss_sppo = (r_w - target).pow(2).mean() + (r_l + target).pow(2).mean()
    h_w = get_hidden_state(policy, w_tok['input_ids'], w_tok['attention_mask'])
    h_l = get_hidden_state(policy, l_tok['input_ids'], l_tok['attention_mask'])
    loss_rep = (h_w * h_l).sum(dim=-1).mean()
    return loss_sppo + lamb * loss_rep, loss_sppo.detach(), loss_rep.detach()
