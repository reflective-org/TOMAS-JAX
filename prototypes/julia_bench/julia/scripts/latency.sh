#!/bin/zsh
# First-call latency in fresh processes: (a) no GasBench precompile workload, (b) with the
# workload (default), (c) PackageCompiler system image. Every run bounded.
#   scripts/latency.sh <log> <tag>
HERE=${0:A:h}; ROOT=${HERE:h}
source $ROOT/env.sh
LOG=$1; TAG=${2:-lat}
CASES=("lim726_L3 900" "syn5742_L3 900")
SOLVERS=(rodas5p fbdf cvode)
# (a) workload off
printf '[GasBench]\nprecompile_workload = false\n' > $ROOT/LocalPreferences.toml
gtimeout -k 20 900 $JULIA --project=$ROOT --startup-file=no -e 'using GasBench' >> $LOG 2>&1
echo "precompile (workload off) exit $?" >> $LOG
for c in $CASES; do for s in $SOLVERS; do
  GASBENCH_WORKLOAD_LABEL=off $HERE/run_case.sh 900 ${=c} $s 1e-3 1e4 1 ${TAG}-off 600 >> $LOG 2>&1
done; done
rm -f $ROOT/LocalPreferences.toml
gtimeout -k 20 900 $JULIA --project=$ROOT --startup-file=no -e 'using GasBench' >> $LOG 2>&1
echo "precompile (workload on) exit $?" >> $LOG
# (b) and (c) interleaved
for c in $CASES; do for s in $SOLVERS; do
  GASBENCH_WORKLOAD_LABEL=on $HERE/run_case.sh 900 ${=c} $s 1e-3 1e4 1 ${TAG}-on 600 >> $LOG 2>&1
  GASBENCH_WORKLOAD_LABEL=sysimage GASBENCH_SYSIMAGE=$ROOT/sysimage/gasbench.dylib GASBENCH_PROJECT=$ROOT/sysimage/env \
    $HERE/run_case.sh 900 ${=c} $s 1e-3 1e4 1 ${TAG}-sysimg 600 >> $LOG 2>&1
done; done
echo "LATENCY DONE $(date)" >> $LOG
