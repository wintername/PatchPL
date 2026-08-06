#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════╗
║  S-SPPO v9 完整代码 — 逐行讲解                              ║
║  Semantic-Calibrated Self-Play Preference Optimization     ║
║  基于 SPPO (Wu et al., ICLR 2025)                          ║
╚══════════════════════════════════════════════════════════════╝

整体流程:
  SFT基座模型 → 自博弈生成6个回复 → 奖励函数打分
  → 选最高分(Winner)和最低分(Loser) → SPPO损失训练LoRA
  → EMA更新参考模型 → 保存 → 下一轮迭代 ×3

核心超参:
  K=6    每题生成6个回复(6个不同prompt模板)
  T=1.0  生成温度(低→稳定, 高→多样)
  λ=1.0  表征校准强度(防止winner/loser特征坍缩)
  β=1e-3 SPPO损失温度系数
  lr=5e-7 LoRA学习率(只训0.3%参数)
"""

# ========== 1. 依赖导入 ==========
import os              # 文件系统操作（建目录、保存模型）
import json            # 保存metrics.json训练日志
import re              # 正则表达式（提取答案、解析计算步骤）
import random          # 打乱训练对顺序
import gc              # 垃圾回收（训练后释放GPU显存）
import math            # 数学函数（exp, sqrt等）

import numpy as np     # 数值计算（计算均值、标准差等统计量）
import torch           # PyTorch深度学习框架
import torch.nn.functional as F  # 激活函数、损失函数（log_softmax, normalize等）

from transformers import (
    AutoModelForCausalLM,           # 自动加载因果语言模型(Qwen2.5-1.5B)
    AutoTokenizer,                  # 自动加载分词器
    get_linear_schedule_with_warmup # 带warmup的线性学习率衰减
)
from datasets import load_dataset   # 从HuggingFace加载GSM8K数据集
from peft import LoraConfig, get_peft_model  # LoRA高效微调(只训0.3%参数)
from tqdm import tqdm              # 进度条


# ========== 2. 路径和硬件配置 ==========

MODEL_PATH = "./deepseek_r1_output/stage1_sft_15b/model"
# ↑ SFT基座模型路径
#   来源: train_r1_method3.py训练(3 epochs,全量7473条GSM8K)
#   架构: Qwen2.5-1.5B (28层, hidden=1536, 12注意力头)
#   GSM8K准确率: 56.71% (748/1319)

OUTPUT_DIR = "./deepseek_r1_output/ssppo_15b_v9"
# ↑ 训练输出目录
#   每轮保存到 iter1/ iter2/ iter3/
#   最终模型保存到该目录根下

DEVICE = "cuda:0"
# ↑ 策略模型(Policy)所在GPU
#   Policy = 基座 + LoRA适配器(可训练)
#   负责: 生成回复 + 前向传播 + 反向传播

DEVICE_REF = "cuda:1"
# ↑ 参考模型(Reference)所在GPU
#   Reference = 基座模型(冻结,不参与训练)
#   负责: 提供KL约束的基线对数概率
#   双GPU设计: Policy和Reference互不干扰


# ========== 3. S-SPPO 核心超参 ==========

NUM_ITERATIONS = 3
# ↑ 外层迭代轮数
#   每轮流程: 生成100题×6回复 → 评分选pair → 训练2epoch → EMA更新 → 保存

K = 6
# ↑ 每题生成的回复数量
#   6个不同的prompt模板各采样1次
#   增加多样性(DiVeRSe思路): 不同角度引导推理

BETA = 1e-3
# ↑ SPPO损失中的温度系数
#   作用: Δ = (win_rate - 0.5) / β
#   值越小→目标差距Δ越大→训练信号越激进

LAMBDA_REP = 1.0
# ↑ 表征校准(Representation Calibration)正则系数
#   作用: 总Loss = loss_sppo + λ·cos(h_winner, h_loser)
#   防止winner/loser的hidden state夹角坍缩到0

LR = 5e-7
# ↑ LoRA适配器的学习率
#   极低的学习率因为LoRA参数少(18.5M)且基座已预训练好

EPOCHS_PER_ITER = 2
# ↑ 每轮对training pairs训练的epoch数
#   epoch=1: 模型见过每对1次, epoch=2: 再加强1次

BATCH_SIZE = 1
# ↑ 微批次大小(逐对训练)
#   S-SPPO按pair训练,每个pair独立计算loss

GRAD_ACCUM = 8
# ↑ 梯度累积步数
#   有效batch = BATCH_SIZE × GRAD_ACCUM = 8对
#   节省显存,等效于batch_size=8

PROMPTS_PER_ITER = 100
# ↑ 每轮从GSM8K训练集取的题目数
#   3轮×100题=300题(不重复,轮间数据不同)

MAX_NEW_TOKENS = 256
# ↑ 单条回复最大生成token数
#   GSM8K推理通常100-200词,256足够

GEN_TEMPERATURE = 1.0
# ↑ 生成温度参数
#   T=1.0: 标准softmax,适度探索
#   T>1.0: 更随机,更多样但可能更差
#   T<1.0: 更确定,更保守

RMS_ENABLED = False
# ↑ 关闭v8的12因子RMs评分器
#   v9只用3因子简化评分


# ========== 4. 模型加载 ==========

print("=" * 60)
print("  S-SPPO: Semantic-Calibrated Self-Play Preference Optimization")
print("=" * 60)

print("\n[1/4] Loading models...")

policy = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,                      # 从本地路径加载
    torch_dtype=torch.bfloat16,      # bfloat16精度(节省显存,RTX3090支持)
    device_map=DEVICE                # 自动放到cuda:0
)
# ↑ Policy模型: 基座Qwen2.5-1.5B + 即将添加的LoRA适配器
#   训练时会更新LoRA参数,基座参数冻结

lora_config = LoraConfig(
    r=16,                            # LoRA秩: 低秩矩阵的维度
                                     # r越大→表达能力越强→但参数越多
    lora_alpha=32,                   # LoRA缩放系数: α/r控制更新幅度
    lora_dropout=0.05,               # Dropout防止过拟合
    target_modules=[                 # 注入LoRA的目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层的QKV和输出投影
        "gate_proj", "up_proj", "down_proj"       # FFN层的门控和投影
    ],
    task_type="CAUSAL_LM"            # 因果语言模型任务
)
# ↑ LoRA原理: 在原始权重旁添加低秩矩阵 ΔW = A·B (A∈R^{d×r}, B∈R^{r×d})
#   原始输出: h = W·x
#   LoRA输出: h = W·x + (α/r)·A·B·x
#   只训练A和B, W保持冻结

policy = get_peft_model(policy, lora_config)
# ↑ 将LoRA配置注入Policy模型
#   注入后可训参数: ~18.5M (原模型1.5B的0.3%)

policy.gradient_checkpointing_enable()
# ↑ 梯度检查点: 用计算换显存
#   前向时不保存全部中间激活,反向时重新计算
#   节省~30%显存, 增加~20%计算时间

print(f"   LoRA trainable params: {sum(p.numel() for p in policy.parameters() if p.requires_grad)/1e6:.1f}M")
# 输出: LoRA trainable params: 18.5M (验证只训练了18.5M参数)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# ↑ 加载分词器: 将文本转为token id序列
#   Qwen2.5的tokenizer使用BPE(Byte-Pair Encoding)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# ↑ 如果分词器没有pad_token, 用eos_token(<|endoftext|>)代替
#   批处理时需要pad_token统一长度

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_REF               # 放到cuda:1
)
# ↑ 参考模型: 基座模型的完整副本, 参数完全冻结
#   作用: 计算KL约束 log π_ref(y|x), 防止策略模型偏离太远

for p in ref_model.parameters():
    p.requires_grad = False
# ↑ 显式冻结所有参数, 确保不参与训练

ref_model.eval()
# ↑ 设为评估模式: 关闭Dropout等训练特有行为

print("   All models loaded! (semantic sim: token-overlap, no network needed)")
# ↑ 语义相似度使用本地token-overlap算法, 不需要SentenceTransformer网络


# ========== 5. 答案提取工具函数 ==========

def extract_answer_math(text):
    """
    从模型回复中提取最终数学答案
    
    提取策略优先级:
      1. \boxed{...}        ← LaTeX数学盒子(最标准)
      2. answer is/:= X     ← 自然语言声明
      3. 返回None           ← 无法提取(后续由normalize处理)
    
    注意: 这是v9的原始版本! 模型实际输出<<9*2=18>>18格式,
          但此函数不识别,导致大量正确答案被误判。
          v10中已修复此问题。
    """
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    # ↑ 正则匹配 \boxed{内容}
    #   r'\\boxed\{' 匹配字面的"\boxed{"
    #   ([^}]*)      捕获任意非}字符
    if m: return m.group(1).strip()
    # ↑ 捕获组1是{}内的内容, strip去掉首尾空白
    
    m = re.search(
        r'(?:answer\s*(?:is|:|=))\s*(\d[\d,.\/]*)\b',
        text, re.IGNORECASE
    )
    # ↑ 匹配 "answer is 18" 或 "answer: 18" 等模式
    #   (?:...) 非捕获组,只分组不保存
    #   \s*      0或多个空白
    #   \d[\d,.\/]* 数字开头,后可跟数字/逗号/点/斜杠
    #   \b       单词边界
    if m: return m.group(1).strip()
    
    return None
    # ↑ 无法提取→返回None
    #   后续normalize(None)→None, 与GT比较→判为错误


def normalize(s):
    """
    答案标准化: 去除格式差异,统一比较
    
    标准化操作:
      去空格, 去逗号(1,000→1000)
      去百分号, 转小写
      去末尾句号
      分数转小数(1/2→0.5)
    
    None安全: 输入None→返回None
    """
    if s is None: return None
    # ↑ 空值保护
    
    s = s.replace(' ', '').replace(',', '').replace('%', '').lower().rstrip('.')
    # ↑ 链式替换: 空格→无, 逗号→无, %→无, 大写→小写, 末尾.→无
    #   例: "1,000.50" → "100050"
    
    if '/' in s:
        # ↑ 检测分数: 如 "1/2", "3/4"
        try:
            parts = s.split('/')
            return str(float(parts[0]) / float(parts[1]))
            # ↑ 分子÷分母, 转回字符串
            #   例: "1/2" → 0.5 → "0.5"
        except:
            pass  # 解析失败→保持原样
    
    return s


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
