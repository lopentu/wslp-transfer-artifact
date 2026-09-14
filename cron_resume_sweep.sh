#!/usr/bin/env bash
#
# Re-queues run_fold_stability.sh for the day's 09:00 window. Added 8/17 after a
# queue that was only ever re-launched by hand lost a whole day: the 8/16 sweep
# deferred 4 cells at 19:46 and nothing ran on 8/17 until 12:34.
#
# Fires at 08:50, not 09:00, on purpose. run_fold_stability.sh resolves its
# window with `date -d "today $EARLIEST"` and, by design, retargets *tomorrow*
# once that instant has passed -- so it never takes the card during the
# colleague's evening. The same rule means a launch at 09:00:00 parks for a full
# 24 hours instead of taking the morning. Firing ten minutes early keeps 09:00
# in the future. Nothing touches the GPU during those ten minutes: the script
# only sleeps until EARLIEST before it so much as reads nvidia-smi.
#
# Launches nothing when a sweep is already up, and nothing when every cell is
# trained, so this stays harmless in the crontab after the queue drains.
# Deadlines are left at the script's own defaults (plan-by 20:00, hard stop
# 20:45) -- an evening released by the colleague is a one-off and belongs on the
# command line that day, not in a recurring job.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin

cd /home/kevin0101/wslp-2026 || exit 1

pgrep -f '[r]un_fold_stability\.sh' >/dev/null && exit 0
DRY=1 ./run_fold_stability.sh 2>/dev/null | grep -q 'to train' || exit 0

log="runs/logs/fold_stability_$(date +%m%d).log"
printf '\n### cron re-queue %s\n' "$(date '+%F %H:%M')" >> "$log"
nohup ./run_fold_stability.sh >> "$log" 2>&1 &
