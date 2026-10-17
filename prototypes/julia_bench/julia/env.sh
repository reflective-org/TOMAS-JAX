# Source this file: sets up the local Julia install and depot (nothing global).
JW=/Users/ali/Documents/GitHub/aerosol-microphysics/tomas-aaar/_work/julia
export JULIA_HOME_LOCAL=$JW/opt/julia-1.12.7
export JULIA=$JULIA_HOME_LOCAL/bin/julia
export JULIA_DEPOT_PATH=$JW/depot:$JULIA_HOME_LOCAL/local/share/julia:$JULIA_HOME_LOCAL/share/julia
export JULIA_PROJECT=$JW/julia-spike
export JULIA_NUM_PRECOMPILE_TASKS=8
export JULIA_PKG_PRECOMPILE_AUTO=1
export BENCH=$JW/bench
export PY=$JW/venv/bin/python
