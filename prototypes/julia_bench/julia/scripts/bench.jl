# One benchmark case in one fresh process, timed per BENCHMARK.md section 7.
#
#     GASBENCH_T0=<epoch seconds at process launch> julia --project=. -t 1 scripts/bench.jl \
#         <problem> <coupling_s> <solver> [rtol] [atol] [n_warm] [tag] [cap_s] [carry_dt]
#
# Writes results/<tag>_<solver>_<problem>_dt<coupling>.json + _hourly.npz (reference format,
# first run) and ..._timing.json (every timing, load, work, memory, accuracy).
const T_LAUNCH = parse(Float64, get(ENV, "GASBENCH_T0", string(time())))
const T_PROC = time()
using GasBench
using JSON3, Statistics, Printf, LinearAlgebra
const T_LOADED = time()

uptime() = strip(read(`uptime`, String))
loadavg() = (l = Sys.loadavg(); [l[1], l[2], l[3]])

problem = ARGS[1]
coupling = parse(Int, ARGS[2])
solver = ARGS[3]
P = load_problem(problem)
rtol = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : P.rtol
atol = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : P.atol
n_warm = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
tag = length(ARGS) >= 7 ? ARGS[7] : "julia"
cap_s = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 1200.0
carry_dt = length(ARGS) >= 9 ? ARGS[9] == "carry" : false
BLAS.set_num_threads(1)

chem = Chem(P.mech)
S = make_solver(solver, chem, P, coupling; rtol = rtol, atol = atol, carry_dt = carry_dt)
ref = GasBench.load_reference(problem, coupling)
const T_SETUP = time()

runs = Any[]
function timed_run(label)
    l0 = loadavg(); u0 = uptime()
    gc0 = Base.gc_num()
    R = run24(S, P, coupling; cap_s = cap_s)
    l1 = loadavg()
    push!(runs, Dict("label" => label, "wall_s" => R.wall_s, "first_interval_s" => R.first_interval_s,
                     "intervals_done" => R.intervals_done, "capped" => R.capped,
                     "load_before" => l0, "load_after" => l1, "uptime_before" => u0,
                     "stats" => R.stats))
    R
end

R1 = timed_run("first")
km = GasBench.key_metric(R1, P, ref)
eall = R1.capped ? NaN : GasBench.all_species_metric(R1, problem, coupling)
warm = Float64[]
if !R1.capped
    for w in 1:n_warm
        Rw = timed_run("warm$w")
        push!(warm, Rw.wall_s)
        Rw.stats == R1.stats || @warn "work differs between runs" R1.stats Rw.stats
    end
end
nint = round(Int, P.duration_s / coupling)
outdir = joinpath(GasBench.bench_dir(), "results")
mkpath(outdir)
rt = @sprintf("%.0e", rtol); at = @sprintf("%.0e", atol)
base = joinpath(outdir, "$(tag)_$(solver)_rtol$(rt)_atol$(at)_$(problem)_dt$(coupling)")
if !R1.capped
    GasBench.write_run(base, R1, P, coupling, "Julia $(VERSION) $(solver) KLU", rtol, atol)
end
timing = Dict{String,Any}(
    "problem" => problem, "coupling_s" => coupling, "solver" => solver, "rtol" => rtol, "atol" => atol,
    "carry_dt" => carry_dt, "julia" => string(VERSION), "threads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(),
    "sysimage" => unsafe_string(Base.JLOptions().image_file),
    "precompile_workload" => get(ENV, "GASBENCH_WORKLOAD_LABEL", "unknown"),
    "t_launch_to_main_s" => T_PROC - T_LAUNCH, "t_package_load_s" => T_LOADED - T_PROC,
    "t_setup_s" => T_SETUP - T_LAUNCH,
    "t_first_s" => R1.wall_s, "t_first_interval_s" => R1.first_interval_s,
    "t_first_capped_extrapolated_s" => R1.capped ? R1.wall_s * nint / R1.intervals_done : nothing,
    "t_warm_s" => warm, "t_warm_median_s" => isempty(warm) ? nothing : median(warm),
    "work_per_24h" => R1.stats, "jac_nnz" => GasBench.jac_nnz(chem), "n" => P.mech.n,
    "peak_rss_mb" => Sys.maxrss() / 2^20,
    "E_key" => km.max, "E_key_median" => km.median, "E_key_p95" => km.p95,
    "worst_species" => km.worst_species, "E_all_hourly" => eall,
    "runs" => runs,
)
open(base * "_timing.json", "w") do io
    JSON3.write(io, timing)
end
@printf("RESULT %s %s dt=%d %s rtol=%.0e atol=%.0e | setup %.2f s | first %.2f s (first interval %.2f s)%s | warm %s median %s | steps %d rej %d nf %d njac %d nw %d | RSS %.0f MB | E_key %.2e (%s) E_all %.2e | load %s\n",
        problem, solver, coupling, carry_dt ? "carry" : "restart", rtol, atol, T_SETUP - T_LAUNCH,
        R1.wall_s, R1.first_interval_s, R1.capped ? " CAPPED after $(R1.intervals_done) intervals" : "",
        join([@sprintf("%.2f", w) for w in warm], ","), isempty(warm) ? "-" : @sprintf("%.2f", median(warm)),
        get(R1.stats, "naccept", -1), get(R1.stats, "nreject", -1), get(R1.stats, "nf", -1),
        get(R1.stats, "njacs", -1), get(R1.stats, "nw", -1), Sys.maxrss() / 2^20, km.max,
        km.worst_species, eall, join([@sprintf("%.1f", r["load_before"][1]) for r in runs], ","))
