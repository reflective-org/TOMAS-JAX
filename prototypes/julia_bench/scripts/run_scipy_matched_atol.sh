#!/bin/zsh
# scipy BDF sparse baseline with atol 1e2 (100x tighter than nominal) at rtol 1e-4 and 1e-3.
export PYTHONDONTWRITEBYTECODE=1
PY=../../venv/bin/python
for rt in 1e-4 1e-3; do
  for p in "lim726_L2 900" "lim726_L3 900" "syn1353_L3 900" "syn4488_L3 900" "syn5742_L3 900" \
           "lim726_L2 60" "lim726_L3 60" "syn1353_L3 60" "syn4488_L3 60" "syn5742_L3 60"; do
    echo "=== $p rtol $rt atol 1e2 | $(uptime)"
    gtimeout 2400 $PY reference.py ${=p} --rtol $rt --atol 1e2 --tag scipysparse_rtol${rt}_atol1e2 --max-wall-s 2300 --out-dir ../results 2>&1 | tail -1
    echo "exit ${pipestatus[1]} $p $rt"
  done
done
