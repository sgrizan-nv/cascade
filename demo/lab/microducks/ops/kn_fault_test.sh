#!/usr/bin/env bash
# Fault injection for the watchdog (run as root, finite): freeze the OVRTX cameras, then the simulation, then kill
# the lab app; after each, wait until the watchdog has restarted it and everything answers again. Log: logs/fault_test.log
L=/home/ubuntu/keynote_night/logs/fault_test.log
say() { echo "$(date -u +%T) $*" | tee -a $L; }
pid_of() { systemctl show -p MainPID --value "$1"; }
wait_ok() {  # url, max seconds
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $2 ]; do
    curl -s -m 5 -o /dev/null -w "%{http_code}" "$1" 2>/dev/null | grep -q 200 && { echo $(( $(date +%s) - t0 )); return 0; }
    sleep 3
  done
  echo timeout; return 1
}
say "fault test start"
for unit in microducks-keynote-ovrtx microducks-keynote-sim; do
  p=$(pid_of $unit)
  say "freeze $unit (SIGSTOP pid $p)"
  t0=$(date +%s)
  kill -STOP "$p"
  for i in $(seq 1 100); do
    np=$(pid_of $unit)
    [ "$np" != "$p" ] && [ "$np" != "0" ] && break
    sleep 3
  done
  say "$unit restarted by the watchdog after $(( $(date +%s) - t0 )) s (new pid $(pid_of $unit))"
  if [ $unit = microducks-keynote-ovrtx ]; then
    say "cameras back after $(wait_ok http://127.0.0.1:18210/snapshot/overview.jpg 180) s more"
  else
    for i in $(seq 1 80); do
      curl -s -m 4 127.0.0.1:18211/api/health | grep -q '"map_ready": true' && break; sleep 3
    done
    say "simulation mapped again after $(( $(date +%s) - t0 )) s from the freeze"
    sleep 20
    say "cameras: $(curl -s -m 5 127.0.0.1:18210/api/health)"
  fi
  sleep 30
done
p=$(pid_of cascade-lab-app)
say "kill -9 cascade-lab-app pid $p (systemd should bring it back by itself)"
kill -9 "$p"
say "lab back after $(wait_ok http://127.0.0.1:8095/healthz 120) s"
say "fault test end"
