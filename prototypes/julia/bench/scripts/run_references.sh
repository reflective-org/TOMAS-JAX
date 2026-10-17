#!/bin/zsh
# Reference solutions (BDF rtol 1e-10, atol 1e-4) and Radau cross-checks; each run bounded.
export PYTHONDONTWRITEBYTECODE=1
PY=../../venv/bin/python
for p in "lim726_L3 900" "lim726_L3 60" "lim726_L2 900" "lim726_L2 60"; do
  gtimeout 3000 $PY reference.py ${=p} --method Radau --rtol 1e-9 --atol 1e-3 --tag chkradau --max-wall-s 2900 2>&1 | tail -1; echo "exit ${pipestatus[1]} $p radau"
done
for p in "syn1353_L3 900" "syn4488_L3 900" "syn5742_L3 900" "syn1353_L3 60" "syn4488_L3 60" "syn5742_L3 60"; do
  gtimeout 3600 $PY reference.py ${=p} --rtol 1e-10 --atol 1e-4 --tag ref --max-wall-s 3500 2>&1 | tail -1; echo "exit ${pipestatus[1]} $p ref"
done
