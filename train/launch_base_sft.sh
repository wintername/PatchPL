#!/bin/bash
# Launch base-model tree SFT via DeepSpeed launcher (ASCII only)
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/wcx/miniconda3/lib:${LD_LIBRARY_PATH:-}
pkill -f "train_base_sf"t 2>/dev/null
sleep 2
mv -f logs/base_sft.log logs/base_sft.log.prev 2>/dev/null
nohup nice -n 10 python -u -m deepspeed.launcher.launch \
  --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMl19 --master_addr=127.0.0.1 --master_port=29501 \
  train_base_sft.py > logs/base_sft.log 2>&1 &
echo "BASE-SFT-STARTED pid=$!"
