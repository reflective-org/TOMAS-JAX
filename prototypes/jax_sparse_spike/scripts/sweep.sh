#!/bin/zsh
# usage: sweep.sh <logfile> <tag> <warm> "<problem coupling extra args>" ...
# one fresh process per line, each bounded by gtimeout; prints a compact summary line
export PYTHONDONTWRITEBYTECODE=1
LOG=$1; TAG=$2; WARM=$3; shift 3
PY=../../venv/bin/python
for args in "$@"; do
  up=$(uptime)
  line=$(gtimeout 1800 $PY run_jax_sparse.py --warm $WARM --tag $TAG ${=args} 2>&1 | grep -v "^W1017" | tail -1)
  rc=${pipestatus[1]}
  summ=$(print -r -- "$line" | python3 -c "
import json,sys
try:
  d=json.loads(sys.stdin.read()); s=d['settings']
  print(f\"{d['problem']} {d['coupling_s']} {s['method']} rtol={s['rtol']:g} atol={s['atol']:g} h0={s['h0']} norm={s.get('norm','rms')} mode={s['mode']} tail={s['tail']} | first={d['t_first_s']:.2f} fi={d['t_first_interval_s']:.2f} warm={d['warm']} steps={d['work']['accepted']} rej={d['work']['rejected']} Ekey={d.get('key_max_err',float('nan')):.2e} ({d.get('key_worst')}) Eall={d.get('all_species_hourly_max') or float('nan'):.2e} rss={d['peak_rss_mb']:.0f} status={d['status']}\")
except Exception as e:
  print('PARSE-FAIL', repr(e))
" 2>&1)
  echo "$summ | exit $rc | $up" | tee -a $LOG
done
