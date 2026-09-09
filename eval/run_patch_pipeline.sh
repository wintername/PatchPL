#!/bin/bash
# Full patch-bridge pipeline: data -> train x2 -> gen -> official harness x2
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/wcx/miniconda3/lib:${LD_LIBRARY_PATH:-}
WORLD=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMl19
L=logs/patch_pipeline.log
ts() { date '+%F %T'; }

echo "$(ts) STEP1 build data" | tee -a $L
/home/wcx/miniconda3/bin/python -u build_patch_sft.py 2>&1 | tee -a $L

echo "$(ts) STEP2 train instruct" | tee -a $L
python -u -m deepspeed.launcher.launch --world_info=$WORLD --master_addr=127.0.0.1 --master_port=29502 \
  train_patch_sft.py models/Qwen2.5-Coder-3B-Instruct checkpoints/coder-3b-instruct-patch instruct-patch \
  > logs/patch_sft_instruct.log 2>&1
echo "$(ts) instruct done rc=$?" | tee -a $L

echo "$(ts) STEP3 train base-ds" | tee -a $L
python -u -m deepspeed.launcher.launch --world_info=$WORLD --master_addr=127.0.0.1 --master_port=29503 \
  train_patch_sft.py checkpoints/coder-3b-base-ds checkpoints/coder-3b-base-patch base-patch \
  > logs/patch_sft_base.log 2>&1
echo "$(ts) base done rc=$?" | tee -a $L

echo "$(ts) STEP4 generate predictions" | tee -a $L
/home/wcx/miniconda3/bin/python -u gen_patch_preds.py checkpoints/coder-3b-instruct-patch coder-3b-instruct-patch 2>&1 | tee -a $L
/home/wcx/miniconda3/bin/python -u gen_patch_preds.py checkpoints/coder-3b-base-patch coder-3b-base-patch 2>&1 | tee -a $L

for TAG in coder-3b-instruct-patch coder-3b-base-patch; do
  echo "$(ts) STEP5 harness $TAG" | tee -a $L
  python -u -m swebench.harness.run_evaluation \
    --dataset_name data/swe_verify/swe_verify_full.json \
    --predictions_path predictions/$TAG.json \
    --max_workers 4 --run_id $TAG-official --timeout 1800 \
    > logs/harness_$TAG.log 2>&1
  echo "$(ts) harness $TAG done rc=$?" | tee -a $L
done

echo "$(ts) STEP6 report" | tee -a $L
/home/wcx/miniconda3/bin/python - <<'PYEOF' | tee -a logs/patch_pipeline.log
import json
for tag in ['coder-3b-instruct-patch', 'coder-3b-base-patch']:
    try:
        d = json.load(open('/home/wcx/swe/%s.%s-official.json' % (tag, tag)))
        s, c, r = d['submitted_instances'], d['completed_instances'], d['resolved_instances']
        print('%s: submitted=%d completed=%d resolved=%d (%.1f%%)' % (tag, s, c, r, 100.0*r/s))
        print('  resolved_ids:', d.get('resolved_ids'))
    except Exception as e:
        print('%s: NO-RESULT %s' % (tag, e))
PYEOF
echo "PATCH-PIPELINE-DONE" | tee -a $L
