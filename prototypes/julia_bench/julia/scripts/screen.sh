#!/bin/zsh
# Screen of further solver settings: lower-order Rosenbrock methods, carrying the step size
# across intervals, and CVODE at rtol 1e-5 where it failed the gate at 1e-4. One fresh
# process per run, bounded.
#   scripts/screen.sh <log> <tag>
HERE=${0:A:h}
LOG=$1; TAG=${2:-screen}
export GASBENCH_WORKLOAD_LABEL=on
rc() { $HERE/run_case.sh 1500 "$@" >> $LOG 2>&1 }
for s in rodas4 ros3p rosenbrock23 ros34pw2; do rc lim726_L3 60 $s 1e-3 1e4 3 $TAG 1200; done
for s in rodas5p rodas4 fbdf; do rc lim726_L3 60 $s 1e-3 1e4 3 ${TAG}carry 1200 carry; done
for c in "syn5742_L3 60" "syn5742_L3 900"; do
  for s in ros3p rodas4; do rc ${=c} $s 1e-3 1e4 3 $TAG 1200; done
  for s in rodas5p fbdf; do rc ${=c} $s 1e-3 1e4 3 ${TAG}carry 1200 carry; done
done
for s in rodas4 ros3p; do rc lim726_L2 900 $s 1e-3 1e4 3 $TAG 1200; done
rc lim726_L2 900 rodas5p 1e-3 1e4 3 ${TAG}carry 1200 carry
for c in "lim726_L2 60" "syn4488_L3 60" "syn5742_L3 60"; do rc ${=c} cvode 1e-5 1e2 3 tight3 1200; done
echo "SCREEN DONE $(date)" >> $LOG
