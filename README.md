# PatchPL — SWE-bench Patch Pipeline (3B experiments)

Pipeline to train small models (Qwen2.5-Coder-3B) to fix GitHub issues,
evaluated with the official SWE-bench harness.

## Results summary (verify set, 250 instances, 12 repos)

| Model | submitted | completed (patch applied) | resolved |
|---|---|---|---|
| coder-3b-base-ds (trajectory SFT) | 250 | 16 | 0 |
| coder-3b-instruct-patch (issue->diff SFT) | 250 | 38 | 0 |
| coder-3b-base-patch (traj->diff two-stage) | 250 | 19 | 0 |

Known 3B ceiling on this set: ~0.4% (1/250, with context tricks).

## Pipeline stages

1. `data/build_task_tree.py` — per-task prefix trees of Nemotron trajectories
   (nodes = message states, edges = actions labeled 1/0).
   Outputs: `task_trees.jsonl`, `best_sft.jsonl` (shortest pass trajectory/task).
2. `data/annotate_deepseek.py` — DeepSeek labels every assistant step 0/1
   (8 workers, checkpointed). Output: `deepseek_labels.jsonl`.
3. `data/build_ds_sft.py` — drop 0-labeled steps, chunk to <=32K tokens.
   Output: `deepseek_sft_chunks.jsonl`.
4. `train/train_base_sft.py` — trajectory SFT on Qwen2.5-Coder-3B base
   (ZeRO-3, 3x3090). Output: `checkpoints/coder-3b-base-ds`.
   Result: model did NOT learn the tool-call format (probe: 0 tool calls).
5. `data/build_patch_sft.py` + `train/train_patch_sft.py` — issue->gold-patch
   SFT (same prompt as evaluation, per advisor's rule).
6. `eval/gen_patch_preds.py` — greedy patch generation.
7. `eval/run_patch_pipeline.sh` — full auto pipeline: data -> train x2 ->
   predictions -> official swebench harness -> report.

## Key lessons

- Trajectory SFT on 3B base fails to learn tool-call format (loss diluted by
  tool outputs; probe shows 0 `<tool_call>` in generation).
- Issue->patch SFT learns the diff FORMAT (38/250 patches apply) but not the
  FIX CONTENT (0 resolved): 250 training pairs are too few for 3B to learn
  real bug fixes.
- Advisor's rule: evaluation prompt must be identical to training prompt.

## Next steps (priority order)

1. Larger model (7B+) — 3B is at its ceiling.
2. Context injection (retrieval of relevant files into the prompt).
3. Much more patch data (official SWE-bench train ~19K).
