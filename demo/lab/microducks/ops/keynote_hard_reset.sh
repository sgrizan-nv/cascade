#!/usr/bin/env bash
# Hard reset of the Keynote tab: restart the simulation and the OVRTX cameras (and Nemotron when it does not answer,
# or always with --nemotron), then wait until all of them work again. Run as root through systemd-run (lab app button
# or by hand). Progress for the web page: state/hard_reset.json. Only one at a time (flock).
#   sudo systemd-run --unit keynote-hard-reset-$(date +%s) --collect /bin/bash /home/ubuntu/keynote_night/ops/keynote_hard_reset.sh
set -u
K=/home/ubuntu/keynote_night
S=$K/state
LOG=$K/logs/hard_reset.log
mkdir -p "$S" "$K/logs"
exec 9>"$S/hard_reset.lock"
if ! flock -n 9; then echo "$(date -u +%FT%TZ) already running" >> "$LOG"; exit 0; fi
FORCE_NEMO=0
[ "${1:-}" = "--nemotron" ] && FORCE_NEMO=1
T0=$(date +%s)
STEPS=""
status() {  # phase message done ok
  local now; now=$(date +%s)
  python3 - "$S/hard_reset.json" "$1" "$2" "$3" "$4" "$T0" "$now" "$STEPS" <<'EOF'
import json, os, sys
p, phase, msg, done, ok, t0, now, steps = sys.argv[1:9]
rec = dict(phase=phase, message=msg, done=done == "1", ok=ok == "1", started=int(t0), updated=int(now),
           elapsed_s=int(now) - int(t0), steps=[s for s in steps.split("|") if s])
tmp = p + ".tmp"
with open(tmp, "w") as fh:
    json.dump(rec, fh)
os.chmod(tmp, 0o644)
os.replace(tmp, p)
EOF
  echo "$(date -u +%FT%TZ) [$1] $2" >> "$LOG"
}
step() { STEPS="${STEPS}|$1"; }
nemo_ok() {
  curl -s -m 25 -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" \
    -d '{"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":"Say OK."}],"max_tokens":2,"chat_template_kwargs":{"enable_thinking":false}}' \
    http://127.0.0.1:18310/v1/chat/completions 2>/dev/null | grep -q 200
}
sim_ready() {
  curl -s -m 4 http://127.0.0.1:18211/api/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('map_ready') and d.get('seq',0)>50 else 1)" 2>/dev/null
}
cam_ready() {
  curl -s -m 10 -o /tmp/keynote_hr_snap.jpg -w "%{http_code}" "http://127.0.0.1:18210/snapshot/overview.jpg" 2>/dev/null | grep -q 200 &&
    [ "$(stat -c %s /tmp/keynote_hr_snap.jpg 2>/dev/null || echo 0)" -gt 5000 ]
}

status starting "Hard reset started" 0 0
timeout 20 nvidia-smi -L > /dev/null 2>&1 && step "GPUs answer" || step "GPU check timed out"

# 1. Nemotron: restart only when it does not answer (or when asked)
NEMO_RESTARTED=0
if [ $FORCE_NEMO = 1 ] || ! nemo_ok; then
  status nemotron "Restarting Nemotron (about 5 min)" 0 0
  systemctl restart microducks-nemotron-vllm
  NEMO_RESTARTED=1
else
  step "Nemotron answers"
fi

# 2. simulation and cameras together (the camera server follows the new pose file by itself)
status restarting "Restarting the simulation and the cameras" 0 0
systemctl restart microducks-keynote-sim
systemctl restart microducks-keynote-ovrtx
step "Simulation and cameras restarted"

ok_sim=0; ok_cam=0
for i in $(seq 1 120); do
  [ $ok_sim = 0 ] && sim_ready && { ok_sim=1; step "Simulation running, cameras mapped the yard"; }
  [ $ok_cam = 0 ] && cam_ready && { ok_cam=1; step "OVRTX cameras streaming"; }
  [ $ok_sim = 1 ] && [ $ok_cam = 1 ] && break
  msg="Waiting for"
  [ $ok_sim = 0 ] && msg="$msg the simulation"
  [ $ok_sim = 0 ] && [ $ok_cam = 0 ] && msg="$msg and"
  [ $ok_cam = 0 ] && msg="$msg the cameras"
  status waiting "$msg ($(( $(date +%s) - T0 )) s)" 0 0
  sleep 3
done

if [ $NEMO_RESTARTED = 1 ]; then
  for i in $(seq 1 180); do
    nemo_ok && { step "Nemotron restarted and answers"; break; }
    status nemotron "Waiting for Nemotron to load ($(( $(date +%s) - T0 )) s)" 0 0
    sleep 5
  done
fi

okn=0; nemo_ok && okn=1
if [ $ok_sim = 1 ] && [ $ok_cam = 1 ] && [ $okn = 1 ]; then
  status done "Everything is back. Ready for orders." 1 1
  python3 /home/ubuntu/cascade-lab/ops/tg_send_local.py "Keynote demo: hard reset done in $(( $(date +%s) - T0 )) s, all OK." > /dev/null 2>&1 || true
else
  status done "Hard reset finished with problems: sim $ok_sim, cameras $ok_cam, Nemotron $okn. Try again." 1 0
  python3 /home/ubuntu/cascade-lab/ops/tg_send_local.py "Keynote demo: hard reset FINISHED WITH PROBLEMS (sim $ok_sim, cameras $ok_cam, Nemotron $okn)." > /dev/null 2>&1 || true
fi
