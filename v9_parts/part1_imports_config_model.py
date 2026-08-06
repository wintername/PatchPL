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


