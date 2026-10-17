#!/bin/zsh
# One timed benchmark case in a fresh Julia process, bounded by a wall-clock timeout.
#   scripts/run_case.sh <timeout_s> <problem> <coupling> <solver> [rtol] [atol] [n_warm] [tag] [cap_s] [carry]
# Env: GASBENCH_SYSIMAGE=<path> to start Julia with a PackageCompiler system image.
#      GASBENCH_PROJECT=<env> to run in another environment (the sysimage one).
#      GASBENCH_WORKLOAD_LABEL=<text> recorded in the timing file.
HERE=${0:A:h}
source $HERE/../env.sh
TO=$1; shift
SYSIMG=()
[[ -n "$GASBENCH_SYSIMAGE" ]] && SYSIMG=(--sysimage "$GASBENCH_SYSIMAGE")
PROJ=${GASBENCH_PROJECT:-$JULIA_PROJECT}
echo "=== $* | sysimage=${GASBENCH_SYSIMAGE:-none} | $(uptime)"
export GASBENCH_T0=$(gdate +%s.%N)
gtimeout -k 20 $TO $JULIA $SYSIMG --project=$PROJ -t 1 --startup-file=no $HERE/bench.jl "$@" 2>&1 | grep -E "RESULT|ERROR|Error|error|Warning|warn" | head -20
rc=${pipestatus[1]}
echo "exit $rc | $(uptime)"
exit $rc
