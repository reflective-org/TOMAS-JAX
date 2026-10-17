#!/bin/zsh
# Final timings of the JAX sparse spike with the benchmark protocol (warm 3), interleaved
# with JAX dense and scipy sparse runs on the key cases (A, B, C, A, B, C ...), then the
# remaining cases, then a repeat of the key spike timings. Every run is bounded.
# Config A (tags final, nominal): Ros3, RMS error norm, Hairer initial step every interval
#   (closest to the dense harness); MATCHED[case] = cheapest setting that passes the gate
#   (logs/sweep_tol.log, logs/explore2.log, logs/explore3.log).
# Config B (tag best): Ros3, max error norm, step size carried between intervals, at the
#   nominal tolerances (passes every gate, logs/explore3.log).
cd ${0:A:h}
export PYTHONDONTWRITEBYTECODE=1
PY=../../venv/bin/python
BENCH=../../bench/scripts
LOG=../logs/final_batch.log
typeset -A MATCHED
MATCHED=(
  "lim726_L2 900" 1e-3  "lim726_L3 900" 1e-3  "syn1353_L3 900" 1e-3  "syn4488_L3 900" 1e-3
  "syn5742_L3 900" 5e-4 "lim726_L2 60" 1e-3   "lim726_L3 60" 1e-3    "syn1353_L3 60" 5e-4
  "syn4488_L3 60" "3e-4 --atol 1e2" "syn5742_L3 60" "3e-4 --atol 1e2"
)
# scipy sparse settings that pass the gate (BENCHMARK.md section 8.1)
typeset -A SCIPY
SCIPY=("lim726_L3 900" "1e-5 1e4" "syn5742_L3 900" "1e-4 1e2" "lim726_L3 60" "1e-5 1e4"
       "syn5742_L3 60" "1e-4 1e2")

spike() {  # case tag rtol [extra]
  local c=$1 tag=$2 r=$3; shift 3
  ./sweep.sh $LOG $tag 3 "$c --rtol $r --tag $tag $*"
}
dense() {  # case warm
  local up=$(uptime)
  local line=$(gtimeout 3000 $PY $BENCH/run_jax_dense.py ${=1} --warm $2 --max-wall-s 1200 --tag jaxdense_ab 2>&1 | grep -v "^W1017" | tail -1)
  echo "DENSE $1 | $line | exit ${pipestatus[1]} | $up" | tee -a $LOG
}
scipy() {  # case
  local up=$(uptime) s=(${=SCIPY[$1]})
  local line=$(gtimeout 1800 $PY $BENCH/reference.py ${=1} --rtol $s[1] --atol $s[2] --tag scipy_ab --max-wall-s 1500 --out-dir ../results 2>&1 | tail -1)
  echo "SCIPY $1 rtol=$s[1] atol=$s[2] | $line | exit ${pipestatus[1]} | $up" | tee -a $LOG
}

echo "=== start $(date) | $(uptime)" | tee -a $LOG
# 1. key cases, interleaved
B="--h0 carry --norm max"
for c in "lim726_L3 900" "syn5742_L3 900" "lim726_L3 60" "syn5742_L3 60"; do
  spike "$c" final $MATCHED[$c]
  spike "$c" best 1e-3 ${=B}
  case $c in
    "lim726_L3 900") dense "$c" 3 ;;
    "lim726_L3 60") dense "$c" 1 ;;
  esac
  scipy "$c"
done
dense "syn1353_L3 900" 1
# 2. the other cases at matched accuracy, and every case at nominal tolerances
for c in "lim726_L2 900" "syn1353_L3 900" "syn4488_L3 900" "lim726_L2 60" "syn1353_L3 60" "syn4488_L3 60"; do
  spike "$c" final $MATCHED[$c]
  spike "$c" best 1e-3 ${=B}
done
for c in "lim726_L2 900" "lim726_L3 900" "syn1353_L3 900" "syn4488_L3 900" "syn5742_L3 900" \
         "lim726_L2 60" "lim726_L3 60" "syn1353_L3 60" "syn4488_L3 60" "syn5742_L3 60"; do
  spike "$c" nominal 1e-3
done
# 3. whole day as one jitted lax.scan (config B), then repeat the key spike timings
for c in "lim726_L3 60" "syn5742_L3 60"; do
  spike "$c" best_scan 1e-3 ${=B} --mode scan
done
for c in "lim726_L3 900" "syn5742_L3 900" "lim726_L3 60" "syn5742_L3 60"; do
  spike "$c" final_repeat $MATCHED[$c]
  spike "$c" best_repeat 1e-3 ${=B}
done
echo "=== end $(date) | $(uptime)" | tee -a $LOG
