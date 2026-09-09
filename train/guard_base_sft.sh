#!/bin/bash
# Hourly self-healing guard for base-model SFT (ASCII only)
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/wcx/miniconda3/lib:${LD_LIBRARY_PATH:-}

GUARD_LOG=logs/base_guard.log
STATE=.base_guard_state
MAX_RESTARTS=5

ts() { date '+%F %T'; }

restarts=0; last_steps=-1; last_progress_ts=""
if [ -f "$STATE" ]; then . "$STATE" 2>/dev/null; fi

save_state() {
  printf 'restarts=%s\nlast_steps=%s\nlast_progress_ts=%s\n' \
    "$restarts" "$last_steps" "$last_progress_ts" > "$STATE"
}

train_alive() { pgrep -f "train_base_sf"t > /dev/null 2>&1; }

cur_steps() {
  grep -a -o -E '[0-9]+/[0-9]+' logs/base_sft.log 2>/dev/null | tail -1 | cut -d/ -f1
}

start_train() {
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "$(ts) RESTART-LIMIT-REACHED, manual check needed" >> "$GUARD_LOG"
    return 1
  fi
  restarts=$((restarts+1))
  save_state
  nohup bash launch_base_sft.sh >> "$GUARD_LOG" 2>&1
  echo "$(ts) restarted train (count=$restarts)" >> "$GUARD_LOG"
  last_progress_ts=""
}

while true; do
  if grep -aq "BASE-SFT-DONE" logs/base_sft.log 2>/dev/null; then
    echo "$(ts) DONE, guard exits" >> "$GUARD_LOG"
    break
  fi
  if train_alive; then
    s=$(cur_steps)
    if [ -n "$s" ]; then
      if [ "$s" != "$last_steps" ]; then
        last_steps=$s; last_progress_ts=$(date '+%F %T')
        save_state
      elif [ -n "$last_progress_ts" ]; then
        stalled=$(( $(date +%s) - $(date -d "$last_progress_ts" +%s) ))
        if [ "$stalled" -gt 7200 ]; then
          echo "$(ts) STALLED 2h at step $s, restarting" >> "$GUARD_LOG"
          pkill -f "train_base_sf"t
          sleep 5
          start_train
        fi
      fi
    fi
  else
    echo "$(ts) trainer dead, restarting" >> "$GUARD_LOG"
    start_train
  fi
  sleep 3600
done
