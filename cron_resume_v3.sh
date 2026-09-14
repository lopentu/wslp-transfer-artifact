#!/usr/bin/env bash
#
# Re-queues the v3 review-response sweeps for the day's window. Same purpose as
# cron_resume_sweep.sh, which exists because a queue only ever re-launched by
# hand lost a whole day on 8/17; the submission deadline is 8/20, so there is no
# day left to lose.
#
# Fires at 09:05, not 08:50 like its predecessor, and the difference is
# deliberate. run_fold_stability.sh holds itself until EARLIEST before it reads
# nvidia-smi, so firing early is safe for that one. run_wrongclip.sh has no such
# hold -- it gates on free memory only -- so this script does the waiting instead:
# it will not launch anything until the card has been essentially idle
# (>= QUIET_NEED MiB free) for QUIET_MIN consecutive minutes, which is the same
# test the other sweeps use to tell "their job ended" from "their job is between
# phases". The colleague holds the card 21:00-09:00 by agreement.
#
# Launches nothing if a sweep is already up, and nothing if there is nothing left
# to do, so it is harmless left in the crontab after the queues drain.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
cd /home/kevin0101/wslp-2026 || exit 1

QUIET_MIN=${QUIET_MIN:-5}
QUIET_NEED=${QUIET_NEED:-16000}
# 8/19: the card is ours through the submission, so this is now a crash safety
# net rather than a hand-back protocol -- if a queue dies overnight, the next
# firing restarts it. The idle test stays because a queue that IS running holds
# the card, and this must not start a second one on top of it.
GIVE_UP_MIN=${GIVE_UP_MIN:-180}

log="runs/logs/v3_resume_$(date +%m%d).log"
say() { printf '### %s %s\n' "$(date '+%F %H:%M')" "$*" >> "$log"; }

pgrep -f '[r]un_review_cells2\.sh' >/dev/null && { say "training queue already up"; exit 0; }

# Anything left? Both queues answer that themselves in DRY mode.
train_pending=$(DRY=1 ./run_review_cells3.sh 2>/dev/null | grep -c 'to train')
clip_pending=$(DRY=1 ./run_wrongclip.sh 2>/dev/null | grep -cE '^  f[0-2]_')
[ "${train_pending:-0}" -eq 0 ] && [ "${clip_pending:-0}" -eq 0 ] && { say "nothing pending"; exit 0; }
say "pending: $train_pending training cells, $clip_pending clip cells"

quiet=0; waited=0
while :; do
  [ -f .gpu-hold ] && { say ".gpu-hold present, standing down"; exit 0; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge "$QUIET_NEED" ]; then
    quiet=$(( quiet + 1 ))
    [ "$quiet" -ge "$QUIET_MIN" ] && break
  else
    quiet=0
  fi
  waited=$(( waited + 1 ))
  [ "$waited" -ge "$GIVE_UP_MIN" ] && { say "card never came free in ${GIVE_UP_MIN} min"; exit 0; }
  sleep 60
done
say "card idle ${QUIET_MIN} min straight, launching"

nohup ./run_review_cells3.sh >> "runs/logs/review_cells2.log" 2>&1 &
sleep 120                      # let the trainer claim its 14.6 GB first
nohup ./run_wrongclip.sh >> "runs/logs/wrongclip.log" 2>&1 &
sleep 30
nohup ./run_repr.sh >> "runs/logs/repr.log" 2>&1 &
say "launched training + wrong-clip + representation queues"
