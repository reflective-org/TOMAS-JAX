#!/bin/zsh
# A/B timings in one session on the shared machine: Julia, scipy sparse BDF and JAX dense
# interleaved (A, B, A, B), each in a fresh process and bounded by a timeout.
#   scripts/ab_session.sh <log> <tag>
HERE=${0:A:h}; ROOT=${HERE:h}
source $ROOT/env.sh
LOG=$1; TAG=${2:-ab}
BS=$BENCH/scripts
export PYTHONDONTWRITEBYTECODE=1
scipy() {  # problem coupling rtol atol tag
  echo "=== scipy $* | $(uptime)" >> $LOG
  (cd $BS && gtimeout -k 20 1500 $PY reference.py $1 $2 --rtol $3 --atol $4 --tag $5 --max-wall-s 1400 --out-dir ../results 2>&1 | tail -1) >> $LOG
  echo "exit ${pipestatus[1]} | $(uptime)" >> $LOG
}
jaxd() {  # problem coupling warm tag
  echo "=== jaxdense $* | $(uptime)" >> $LOG
  (cd $BS && gtimeout -k 20 1500 $PY run_jax_dense.py $1 $2 --warm $3 --max-wall-s 1200 --tag $4 2>&1 | grep -v "^W1017" | tail -1) >> $LOG
  echo "exit ${pipestatus[1]} | $(uptime)" >> $LOG
}
jl() {  # problem coupling solver rtol atol tag [carry]
  GASBENCH_WORKLOAD_LABEL=on $HERE/run_case.sh 1500 $1 $2 $3 $4 $5 3 $6 1200 $7 >> $LOG 2>&1
}
for pass in 1 2; do
  jl lim726_L3 900 fbdf 1e-3 1e4 ${TAG}${pass}
  scipy lim726_L3 900 1e-5 1e4 scipysparse_${TAG}${pass}_rtol1e-5
  jaxd lim726_L3 900 3 jaxdense_${TAG}${pass}
  jl syn5742_L3 900 rodas5p 1e-3 1e4 ${TAG}${pass}
  jl syn5742_L3 900 rodas5p 1e-3 1e4 ${TAG}${pass}carry carry
  scipy syn5742_L3 900 1e-4 1e2 scipysparse_${TAG}${pass}_rtol1e-4_atol1e2
  jl syn5742_L3 60 rodas5p 1e-3 1e4 ${TAG}${pass}carry carry
  jl syn5742_L3 60 fbdf 1e-3 1e4 ${TAG}${pass}
done
scipy syn5742_L3 60 1e-4 1e2 scipysparse_${TAG}1_rtol1e-4_atol1e2
echo "AB DONE $(date)" >> $LOG
