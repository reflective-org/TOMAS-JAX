#!/bin/zsh
# Same-session A/B against the sparse JAX spike (a private copy of its code in abjax/, run
# unchanged, outputs in abjax/results): A, B, A, B, each a fresh bounded process.
#   scripts/ab_jaxsparse.sh <log>
HERE=${0:A:h}; ROOT=${HERE:h}
source $ROOT/env.sh
LOG=$1
export PYTHONDONTWRITEBYTECODE=1 TOMAS_GAS_BENCH=$BENCH
jx() {  # problem coupling tag args...
  echo "=== jaxsparse $* | $(uptime)" >> $LOG
  (cd $ROOT/abjax/scripts && gtimeout -k 20 900 $PY run_jax_sparse.py $1 $2 --tag $3 ${@:4} 2>&1 | grep -v "^W1017" | tail -1) >> $LOG
  echo "exit ${pipestatus[1]} | $(uptime)" >> $LOG
}
jl() {  # problem coupling solver rtol atol tag [carry]
  GASBENCH_WORKLOAD_LABEL=on $HERE/run_case.sh 900 $1 $2 $3 $4 $5 3 $6 600 $7 >> $LOG 2>&1
}
for pass in 1 2; do
  jx syn5742_L3 900 abA$pass --rtol 5e-4
  jl syn5742_L3 900 fbdf 1e-4 1e4 abjx$pass
  jl syn5742_L3 900 rodas5p 1e-3 1e4 abjx${pass}carry carry
  jx syn5742_L3 60 abB$pass --h0 carry --norm max
  jl syn5742_L3 60 rodas5p 1e-3 1e4 abjx${pass}carry carry
  jl syn5742_L3 60 ros3p 1e-3 1e4 abjx$pass
  jx lim726_L3 60 abB$pass --h0 carry --norm max
  jl lim726_L3 60 rodas4 1e-3 1e4 abjx${pass}carry carry
done
echo "ABJAX DONE $(date)" >> $LOG
