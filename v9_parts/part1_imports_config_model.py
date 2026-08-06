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
MODEL_PATH = "./deepseek_r1_output/stage1_sft_15b/model"
OUTPUT_DIR = "./deepseek_r1_output/ssppo_15b_v9"
DEVICE = "cuda:0"
DEVICE_REF = "cuda:1"
NUM_ITERATIONS = 3
K = 6
BETA = 1e-3
LAMBDA_REP = 1.0
LR = 5e-7
EPOCHS_PER_ITER = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8
PROMPTS_PER_ITER = 100
MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 1.0
RMS_ENABLED = False
print("=" * 60)
print("  S-SPPO: Semantic-Calibrated Self-Play Preference Optimization")
print("=" * 60)
print("\n[1/4] Loading models...")
policy = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,                      # 从本地路径加载
    torch_dtype=torch.bfloat16,      # bfloat16精度(节省显存,RTX3090支持)
    device_map=DEVICE                # 自动放到cuda:0
)
lora_config = LoraConfig(
    r=16,                            # LoRA秩: 低秩矩阵的维度
    lora_alpha=32,                   # LoRA缩放系数: α/r控制更新幅度
    lora_dropout=0.05,               # Dropout防止过拟合
    target_modules=[                 # 注入LoRA的目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层的QKV和输出投影
        "gate_proj", "up_proj", "down_proj"       # FFN层的门控和投影
    ],
    task_type="CAUSAL_LM"            # 因果语言模型任务
)
policy = get_peft_model(policy, lora_config)
policy.gradient_checkpointing_enable()
print(f"   LoRA trainable params: {sum(p.numel() for p in policy.parameters() if p.requires_grad)/1e6:.1f}M")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_REF               # 放到cuda:1
)
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()
print("   All models loaded! (semantic sim: token-overlap, no network needed)")
