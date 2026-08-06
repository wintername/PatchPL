# ========== 7. 语义相似度 (本地实现, 无网络依赖) ==========

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
    # ↑ 批量处理: 输入是列表→逐个计算→返回tensor
    
    def tokenize(s):
        # 简单分词: 提取所有数字和长度≥2的英文单词
        return set(re.findall(r'\d+|[a-zA-Z]{2,}', s.lower()))
    
    ta = tokenize(texts_a)
    tb = tokenize(texts_b)
    # ↑ 两个文本分别分词,得到token集合
    
    if not ta or not tb:
        return torch.tensor(0.5)
    # ↑ 任一文本无token→无法比较→返回中性值0.5
    
    jaccard = len(ta & tb) / len(ta | tb)
    # ↑ Jaccard相似度: 交集大小/并集大小
    
    len_penalty = min(len(texts_a), len(texts_b)) / max(len(texts_a), len(texts_b), 1)
    # ↑ 长度惩罚: 短文本/长文本
    #   两文本长度差异大→相似度应降低
    
    sim = 0.7 * jaccard + 0.3 * len_penalty
    return torch.tensor(sim)
    # ↑ 加权融合: 70%内容重叠 + 30%长度一致性


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
    # ↑ 裁剪sim到[0,1]范围
    return 0.5 + (1.0 - sim) * (raw_wr - 0.5)
    # ↑ 核心公式: 相似度→信号衰减


# ========== 8. S-SPPO 损失函数 ==========

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
    # ↑ 前向传播,要求输出所有层的hidden states
    
    hidden = outputs.hidden_states[-1]
    # ↑ 取最后一层的hidden states shape: (batch, seq_len, hidden_dim)
    
    last_pos = attention_mask.sum(dim=1) - 1
    # ↑ 每条序列最后一个非padding token的位置
    #   attention_mask: [1,1,1,1,0,0] → sum=4 → last_pos=3
    
    h = hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos]
    # ↑ 索引: hidden[batch_idx, last_position]
    #   提取每条序列最后一个有效token的hidden state
    
    return F.normalize(h.float(), p=2, dim=-1).to(torch.bfloat16)
    # ↑ L2归一化: 使向量模长=1, 方便计算余弦相似度


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
    # ———— 分词 ————
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

    # ———— 策略模型前向 ————
    pol_w_out = policy(**w_tok)
    # ↑ Policy(winner_text) → logits shape: (1, seq_len, vocab_size)
    pol_l_out = policy(**l_tok)
    # ↑ Policy(loser_text)  → 同上
    
    # ———— 参考模型前向(冻结,无梯度) ————
    with torch.no_grad():
        # 把数据移到Ref所在GPU(cuda:1)
        w_tok_ref = {k: v.to(ref_model.device) for k, v in w_tok.items()}
        l_tok_ref = {k: v.to(ref_model.device) for k, v in l_tok.items()}
        
        ref_w_out = ref_model(**w_tok_ref)
        ref_l_out = ref_model(**l_tok_ref)
        # ↑ Ref模型输出logits,不参与训练

    # ———— 计算序列对数概率 ————
    def seq_logp(logits, input_ids):
        """
        计算序列的对数概率: Σ log P(token_i | token_{<i})
        
        原理:
          logits → softmax → log → 取每个位置实际token的概率 → 求和
          只对非padding位置求和
        """
        logp = F.log_softmax(logits, dim=-1)
        # ↑ shape: (batch, seq_len, vocab_size)
        #   每个位置对词表取log概率
        
        labels = input_ids[:, 1:]
        # ↑ 标签 = 输入右移1位
        #   input:  [A, B, C, D]
        #   labels: [B, C, D]    (预测B时看A, 预测C时看AB...)
        
        lp = logp[:, :-1].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        # ↑ 从logp中取每个位置实际token的log概率
        #   gather: 按labels索引收集值
        #   shape: (batch, seq_len-1)
        
        mask = (labels != tokenizer.pad_token_id).float()
        # ↑ padding位置不计算损失
        
        return (lp * mask).sum(dim=-1)
        # ↑ 加权求和: 每条序列的总log概率

    pol_w_lp = seq_logp(pol_w_out.logits, w_tok['input_ids'])
    pol_l_lp = seq_logp(pol_l_out.logits, l_tok['input_ids'])
    # ↑ 策略模型下winner和loser的对数概率
    
    ref_w_lp = seq_logp(ref_w_out.logits, w_tok_ref['input_ids']).to(policy.device)
    ref_l_lp = seq_logp(ref_l_out.logits, l_tok_ref['input_ids']).to(policy.device)
    # ↑ 参考模型下的对数概率(移到Policy GPU便于后续计算)

    r_w = pol_w_lp - ref_w_lp
    # ↑ winner的相对增益
    #   r_w>0 → 策略模型比参考模型更喜欢winner ✓
    r_l = pol_l_lp - ref_l_lp
    # ↑ loser的相对增益
    #   r_l<0 → 策略模型比参考模型更不喜欢loser ✓

    # ———— SPPO目标: 拉大差距 ————
    target = torch.tensor(
        (cal_win_rate_target - 0.5) / bet,
        device=policy.device,
        dtype=torch.bfloat16
    )
    # ↑ Δ = (win_rate - 0.5) / β
    #   例: win_rate=0.8, β=0.001 → Δ=300
    
    loss_sppo = (r_w - target).pow(2).mean() + (r_l + target).pow(2).mean()
    # ↑ (r_w - Δ)²: winner应达到目标增益
    #   (r_l + Δ)²: loser应在目标增益的负方向
    #   → 两者差距 = 2Δ

    # ———— 表征校准: 防止坍缩 ————
    h_w = get_hidden_state(policy, w_tok['input_ids'], w_tok['attention_mask'])
    h_l = get_hidden_state(policy, l_tok['input_ids'], l_tok['attention_mask'])
    # ↑ 提取winner和loser最后一个token的hidden state
    
    loss_rep = (h_w * h_l).sum(dim=-1).mean()
    # ↑ 余弦相似度(因为已L2归一化,点积=cos)
    #   cos→1(坍缩)→loss_rep大→惩罚
    #   cos→0(正交)→loss_rep小→不惩罚

    return loss_sppo + lamb * loss_rep, loss_sppo.detach(), loss_rep.detach()
    # ↑ 返回:
    #   总loss (需要反向传播)
    #   sppo_loss (仅日志)
    #   rep_loss  (仅日志)


