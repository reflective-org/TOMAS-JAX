#!/bin/zsh
# JAX dense timings with the benchmark protocol; one fresh process per line, each bounded.
export PYTHONDONTWRITEBYTECODE=1
PY=../../venv/bin/python
run() {  # problem coupling warm cap tag
  echo "=== $1 $2 warm=$3 cap=$4 tag=$5 | $(uptime)"
  gtimeout $(( $4 * 5 + 600 )) $PY run_jax_dense.py $1 $2 --warm $3 --max-wall-s $4 --tag $5 2>&1 | grep -v "^W1017" | tail -1
  echo "exit ${pipestatus[1]} $1 $2 $5"
}
run lim726_L3 900 3 1200 jaxdense
run lim726_L2 900 3 1200 jaxdense
run syn1353_L3 900 3 1200 jaxdense
run lim726_L3 60 3 1200 jaxdense
run lim726_L2 60 3 1200 jaxdense
run syn1353_L3 60 2 1200 jaxdense
run syn4488_L3 900 0 1200 jaxdense
run syn5742_L3 900 0 1200 jaxdense
# repeat the key timings at the end
run lim726_L3 900 3 1200 jaxdense_repeat
run syn1353_L3 900 3 1200 jaxdense_repeat
run lim726_L3 60 3 1200 jaxdense_repeat
