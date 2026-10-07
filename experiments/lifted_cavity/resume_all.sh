#!/bin/sh
# Relaunch the job queues after a container restart (each step resumes from its checkpoint or is skipped if done).
cd "$(dirname "$0")"
for q in qcore1b qcore2; do
  if ! ps -eo args | grep -v grep | grep -q "^sh $q.sh"; then
    setsid nohup sh $q.sh >> ${q%b}.log 2>&1 < /dev/null &
    echo "relaunched $q"
  fi
done
