#!/usr/bin/env bash
# Poll HPC3 every 5 min; exit (waking the loop) on: a tracked job leaving
# RUNNING/PENDING, an E4 job running >25 min without metrics, or any E4 run
# reaching step 100 for the first time. Args: job ids (comma list).
J="$1"; SEEN_FILE="$2"
while true; do
  O=$(ssh -o BatchMode=yes -o ConnectTimeout=40 HPC3_jhe724 "
    sacct -j $J -X -n -o JobID,JobName%22,State,Elapsed | grep -v -E 'RUNNING|PENDING'
    squeue -u jhe724 -h -t R -o '%i %j %M' | while read id name el; do
      case \$name in q35-e4-*) v=\${name#q35-e4-}; f=/data/user/jhe724/outputs/research_q35/e4_\$v/metrics.jsonl
        if [ -f \$f ]; then echo STEP \$v \$(tail -1 \$f | python3 -c 'import sys,json;print(json.loads(sys.stdin.read())[\"step\"])')
        else echo NOMETRICS \$v \$el; fi;; esac; done" 2>/dev/null)
  if echo "$O" | grep -q -E "FAILED|OUT_OF|CANCEL|TIMEOUT|COMPLETED|NODE_FAIL"; then echo "JOB_STATE_CHANGE"; echo "$O"; exit 0; fi
  if echo "$O" | awk '/NOMETRICS/{split($3,t,":"); n=split($3,t,":"); m=(n==3)?t[1]*60+t[2]:t[1]; if (index($3,"-")>0 || m>=25) f=1} END{exit !f}'; then echo "STALLED_NO_METRICS"; echo "$O"; exit 0; fi
  for v in $(echo "$O" | awk '/^STEP/ && $3>=100 {print $2}'); do
    grep -qx "$v" "$SEEN_FILE" 2>/dev/null || { echo "$v" >> "$SEEN_FILE"; echo "E4_REACHED_100 $v"; echo "$O"; exit 0; }
  done
  sleep 300
done
