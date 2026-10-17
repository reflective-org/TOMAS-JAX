#!/bin/zsh
# Run a list of benchmark cases, one fresh process each, interleaving solvers within a case.
#   scripts/run_batch.sh <log> <tag> <n_warm> "<solver ...>" "<problem coupling [rtol atol]>" ...
# Every run is bounded (timeout 1500 s, in-run cap 1200 s). Exit codes are logged.
HERE=${0:A:h}
LOG=$1; TAG=$2; NW=$3; SOLVERS=(${=4}); shift 4
for c in "$@"; do
  parts=(${=c})
  rt=${parts[3]:-1e-3}; at=${parts[4]:-1e4}
  for s in $SOLVERS; do
    $HERE/run_case.sh 1500 ${parts[1]} ${parts[2]} $s $rt $at $NW $TAG 1200 >> $LOG 2>&1
  done
done
echo "BATCH DONE $(date)" >> $LOG
