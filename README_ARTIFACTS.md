# 数据与结果产物说明(Data & Results Artifacts)

本目录树由导师要求整理:代码 + 全部数据 + 全部结果,便于结合代码分析 resolve=0 的原因。

## 目录结构

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `results/official/` | 4 次官方 SWE-bench harness 评测结果 JSON(submitted/completed/resolved) | 约 110K |
| `results/predictions/` | 各模型生成的预测补丁(输入 harness 的原始文件) | 约 5.4M |
| `results/logs/` | 训练与评测完整日志(base_sft/nemotron_sft/patch_sft/harness/pipeline) | 约 1.3M |
| `results/gpt_verdicts_case.json` | GPT 轨迹判例标注样例 | 3.6K |
| `dataset/swe_verify/` | 自建 500 实例评测集 full/test/train/easy/medium/hard/rl + deepseek_traces + dpo_pairs | 约 49M |
| `dataset/nemotron/` | 轨迹训练数据(best_sft / deepseek_sft / deepseek_sft_chunks / nemotron_labeled / deepseek_labels) | gzip 压缩 |
| `dataset/patch_sft/` | 补丁桥接训练数据(250 条 issue→gold patch) | 832K |
| `dataset/swe_all_train.jsonl.gz` | 合并后的训练语料 | gzip 压缩 |

## 未上传的大文件(服务器本地路径)

- **模型 checkpoint(共 127G)**:`/home/wcx/swe/checkpoints/{coder-3b-base-ds, coder-3b-base-patch, coder-3b-base-tree, coder-3b-instruct-patch, coder-3b-nemotron}`
- **基座模型(5.8G)**:`/home/wcx/swe/models/Qwen2.5-Coder-3B`
- **官方 SWE-bench 原始数据集副本**:`/home/wcx/swe/data/swe-bench_train.jsonl`(126M)、`swe-bench_test.jsonl`(276M)——公开数据,可按需从官方仓库重新下载

## 关键结论速览(详见 README.md)

- 全部 4 轮官方评测 resolved = 0 / 250
- 轨迹 SFT 在 3B 基座上未生效(loss 稀释,模型未学会 `<tool_call>` 格式)
- 补丁桥接 SFT 学会了格式(instruct 38/250 个补丁可 apply)但未学会修复内容
- 结论:3B 容量天花板,建议换 7B / 注入上下文 / 扩数据
