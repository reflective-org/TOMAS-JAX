#!/bin/zsh
# Extra baseline: scipy BDF + SuperLU with the analytic sparse Jacobian at the tutorial
# tolerances (rtol 1e-3, atol 1e4), every case twice (A then B pass), each run bounded.
export PYTHONDONTWRITEBYTECODE=1
PY=../../venv/bin/python
for pass in a b; do
  for p in "lim726_L2 900" "lim726_L3 900" "syn1353_L3 900" "syn4488_L3 900" "syn5742_L3 900" \
           "lim726_L2 60" "lim726_L3 60" "syn1353_L3 60" "syn4488_L3 60" "syn5742_L3 60"; do
    echo "=== $p pass $pass | $(uptime)"
    gtimeout 2400 $PY reference.py ${=p} --rtol 1e-3 --atol 1e4 --tag scipysparse_$pass --max-wall-s 2300 --out-dir ../results 2>&1 | tail -1
    echo "exit ${pipestatus[1]} $p $pass"
  done
done
