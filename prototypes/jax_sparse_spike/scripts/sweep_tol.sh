#!/bin/zsh
# Tolerance sweep (rtol first, atol 1e4) of the default spike solver on all 10 cases.
cd ${0:A:h}
args=()
for case in "lim726_L2 900" "lim726_L3 900" "syn1353_L3 900" "syn4488_L3 900" "syn5742_L3 900" \
            "lim726_L2 60" "lim726_L3 60" "syn1353_L3 60" "syn4488_L3 60" "syn5742_L3 60"; do
  for r in 1e-3 5e-4 2e-4 1e-4; do
    args+=("$case --rtol $r --tag sw_r$r")
  done
done
./sweep.sh ../logs/sweep_tol.log sw 1 "${args[@]}"
