#!/usr/bin/env bash
# Prescribed-J sensitivity on the AFGL tropical altitudes, adaptive gas sub-steps
# (sensitivities/dunne_nuc/prescribed_j_tropical.py, docs/prescribed_j.md).
#
#   sensitivities/dunne_nuc/run_prescribed_j_overnight.sh [BINS] [STAGE]
#
#   BINS   40 (default) or 80
#   STAGE  main  (default) every SO2 up to 10 ppmv: 6 altitudes x 7 levels x 9 J
#          hi    100 ppmv only (the slowest cells; run after main)
#          all   main, then hi
#
# Resumable: files that exist are skipped, so rerunning after an interruption
# continues. Per-file logs sit beside the NPZ files in
# sensitivities/dunne_nuc/results/prescribed_j_tropical/<BINS>bin/.
# NJOBS (default 6, the performance cores) processes run at a time.
#
# Keep the Mac awake. Preferred (also lid closed on battery; needs your sudo
# password, so run it yourself in an interactive shell with the venv active):
#     run_awake -m sensitivities.dunne_nuc.prescribed_j_tropical --all [--bins 80] [--so2 1e5]
# This script uses caffeinate -ims instead, which holds only while plugged in
# and does not stop sleep with the lid closed on battery.
set -euo pipefail
cd "$(dirname "$0")/../.."
BINS="${1:-40}"
STAGE="${2:-main}"
export NJOBS="${NJOBS:-6}"
PY=(python -m sensitivities.dunne_nuc.prescribed_j_tropical --bins "$BINS")
[ -d .venv ] && source .venv/bin/activate

run() { caffeinate -ims "${PY[@]}" "$@"; }

case "$STAGE" in
  main) run --all ;;
  hi)   run --all --so2 1e5 ;;
  all)  run --all; run --all --so2 1e5 ;;
  *) echo "STAGE must be main, hi or all" >&2; exit 2 ;;
esac
echo "done: figures in sensitivities/dunne_nuc/results/prescribed_j_tropical/${BINS}bin/"
