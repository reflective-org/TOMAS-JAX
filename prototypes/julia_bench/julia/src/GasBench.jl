"""
    GasBench

Julia side of the shared gas-chemistry benchmark (`_work/julia/bench/BENCHMARK.md`).

Reads the neutral problem files, builds the right-hand side from index lists and an analytic
sparse Jacobian with a fixed sparsity pattern, and integrates 24 h interval by interval with
the rate constants frozen over each coupling interval (KPP semantics):

    r_j     = k_j * prod over reactant entries a of y_a
    dy_s/dt = sum_j nu_sj r_j
    J_sa    = sum_j nu_sj k_j * y_other(j, a)          (the other reactant entry, or 1)

The solvers are OrdinaryDiffEq (Rosenbrock and BDF families, KLU through LinearSolve) and
Sundials CVODE_BDF with KLU. Each interval is a new initial-value problem; the integrator
object is reused (`reinit!`) so the KLU symbolic analysis is done once.
"""
module GasBench

using JSON3
using SparseArrays
using LinearAlgebra
using Statistics
using Printf
using SciMLBase
using OrdinaryDiffEqCore
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqBDF
using LinearSolve
using Sundials
using NPZ
using PrecompileTools

export load_problem, Chem, rhs!, jac!, make_solver, run24, bench_dir, solver_names

bench_dir() = get(ENV, "BENCH", normpath(joinpath(@__DIR__, "..", "..", "bench")))

# ----------------------------------------------------------------------------------------
# Problem files
# ----------------------------------------------------------------------------------------

struct Mechanism
    name::String
    species::Vector{String}
    n::Int
    nr::Int
    r1::Vector{Int32}      # first reactant entry (1-based)
    r2::Vector{Int32}      # second reactant entry, 0 if none
    S::SparseMatrixCSC{Float64,Int}   # n x nr net stoichiometry nu_sj
    ro2::Vector{Int}
end

struct Problem
    name::String
    mech::Mechanism
    k_thermal::Vector{Float64}
    ro2_dep::BitVector
    Jcoef::SparseMatrixCSC{Float64,Int}   # nr x nJ
    external::Vector{Tuple{Int,Float64}}
    jtab::Dict{Int,Matrix{Float64}}       # coupling -> nJ x n_intervals
    y0::Vector{Float64}
    reset_zero::Vector{Int}
    reset_initial::Vector{Int}
    key_species::Vector{String}
    key_index::Vector{Int}
    rtol::Float64
    atol::Float64
    duration_s::Float64
end

function load_mechanism(path::AbstractString)
    d = JSON3.read(read(path, String))
    n = Int(d.n_species)
    nr = Int(d.n_reactions)
    r1 = zeros(Int32, nr)
    r2 = zeros(Int32, nr)
    I = Int[]; Jr = Int[]; V = Float64[]
    for j in 1:nr
        rs = d.reactants[j]
        1 <= length(rs) <= 2 || error("reaction $j has $(length(rs)) reactant entries")
        r1[j] = Int(rs[1]) + 1
        if length(rs) == 2
            r2[j] = Int(rs[2]) + 1
        end
        for a in rs
            push!(I, Int(a) + 1); push!(Jr, j); push!(V, -1.0)
        end
        for pc in d.products[j]
            push!(I, Int(pc[1]) + 1); push!(Jr, j); push!(V, Float64(pc[2]))
        end
    end
    S = sparse(I, Jr, V, n, nr)       # duplicates are summed
    dropzeros!(S)
    ro2 = [Int(x) + 1 for x in d.ro2]
    Mechanism(String(d.name), [String(s) for s in d.species], n, nr, r1, r2, S, ro2)
end

function load_problem(name::AbstractString; dir::AbstractString = joinpath(bench_dir(), "problems"))
    d = JSON3.read(read(joinpath(dir, "problem_$(name).json"), String))
    Int(get(d, :index_base, 0)) == 0 || error("expected 0-based indices")
    mech = load_mechanism(joinpath(dir, String(d.mechanism_file)))
    rates = d.rates
    nr = mech.nr
    nJ = length(rates.j_names)
    Jcoef = sparse([Int(e[1]) + 1 for e in rates.j_coef], [Int(e[2]) + 1 for e in rates.j_coef],
                   [Float64(e[3]) for e in rates.j_coef], nr, nJ)
    ro2_dep = falses(nr)
    for j in rates.ro2_dependent
        ro2_dep[Int(j)+1] = true
    end
    jtab = Dict{Int,Matrix{Float64}}()
    for c in d.couplings_s
        rows = d.photolysis[Symbol(string(Int(c)))]
        M = zeros(nJ, length(rows))
        for (i, row) in enumerate(rows), q in 1:nJ
            M[q, i] = Float64(row[q])
        end
        jtab[Int(c)] = M
    end
    y0 = zeros(mech.n)
    for e in d.initial.y0_nonzero
        y0[Int(e[1])+1] = Float64(e[2])
    end
    post = d.post_interval
    key = [String(s) for s in d.key_species]
    keyidx = [Int(i) + 1 for i in d.key_species_index]
    for (s, i) in zip(key, keyidx)
        mech.species[i] == s || error("key species index mismatch for $s")
    end
    Problem(String(d.name), mech, Float64.(collect(rates.k_thermal)), ro2_dep, Jcoef,
            [(Int(e.reaction) + 1, Float64(e.k)) for e in rates.external], jtab, y0,
            [Int(i) + 1 for i in post.reset_to_zero], [Int(i) + 1 for i in post.reset_to_initial],
            key, keyidx, Float64(d.tolerances.rtol), Float64(d.tolerances.atol),
            Float64(d.duration_s))
end

n_intervals(P::Problem, coupling::Int) = round(Int, P.duration_s / coupling)

"""Rate constants of one interval: k = k_thermal .* (RO2 or 1) + C_J * J_i, then external."""
function rate_constants!(k::Vector{Float64}, P::Problem, y::AbstractVector, jcol::AbstractVector)
    ro2 = 0.0
    @inbounds for s in P.mech.ro2
        ro2 += y[s]
    end
    @inbounds for j in eachindex(k)
        k[j] = P.ro2_dep[j] ? P.k_thermal[j] * ro2 : P.k_thermal[j]
    end
    mul!(k, P.Jcoef, jcol, 1.0, 1.0)
    for (j, v) in P.external
        k[j] = v
    end
    return ro2
end

# ----------------------------------------------------------------------------------------
# Right-hand side and analytic sparse Jacobian
# ----------------------------------------------------------------------------------------

"""
Chemistry kernel of one mechanism. `Chem` is callable as the in-place right-hand side
`f(du, u, k, t)`; `jac!(J, u, k, t)` fills the analytic Jacobian on the fixed pattern
`Jproto` (which always contains the diagonal, so W = I - gamma J has the same pattern).
"""
struct Chem
    n::Int
    nr::Int
    r1::Vector{Int32}
    r2::Vector{Int32}
    S::SparseMatrixCSC{Float64,Int}
    rate::Vector{Float64}
    jpos::Vector{Int32}       # position in nonzeros(J)
    jrxn::Vector{Int32}       # reaction
    jother::Vector{Int32}     # other reactant entry, 0 if none
    jnu::Vector{Float64}      # nu_sj
    Jproto::SparseMatrixCSC{Float64,Int}
end

function Chem(m::Mechanism)
    rows = Int[]; cols = Int[]; rxn = Int32[]; other = Int32[]; nu = Float64[]
    S = m.S
    for j in 1:m.nr
        for p in nzrange(S, j)
            s = rowvals(S)[p]; v = nonzeros(S)[p]
            for (a, b) in ((m.r1[j], m.r2[j]), (m.r2[j], m.r1[j]))
                a == 0 && continue
                push!(rows, s); push!(cols, a); push!(rxn, j); push!(other, b); push!(nu, v)
            end
        end
    end
    # pattern: all Jacobian entries plus the full diagonal
    pr = vcat(rows, 1:m.n); pc = vcat(cols, 1:m.n)
    Jproto = sparse(pr, pc, zeros(length(pr)), m.n, m.n)
    pos = Vector{Int32}(undef, length(rows))
    for e in eachindex(rows)
        rng = nzrange(Jproto, cols[e])
        q = searchsortedfirst(view(rowvals(Jproto), rng), rows[e])
        pos[e] = first(rng) + q - 1
        rowvals(Jproto)[pos[e]] == rows[e] || error("pattern lookup failed")
    end
    Chem(m.n, m.nr, m.r1, m.r2, m.S, zeros(m.nr), pos, rxn, other, nu, Jproto)
end

"""Number of structural nonzeros of the Jacobian without the added diagonal."""
jac_nnz(c::Chem) = length(unique(c.jpos))

@inline function rates!(r::Vector{Float64}, u::AbstractVector{Float64}, k::AbstractVector{Float64},
                        r1::Vector{Int32}, r2::Vector{Int32})
    @inbounds for j in eachindex(r)
        v = k[j] * u[r1[j]]
        b = r2[j]
        if b != 0
            v *= u[b]
        end
        r[j] = v
    end
    return r
end

function (c::Chem)(du, u, k, t)
    rates!(c.rate, u, k, c.r1, c.r2)
    mul!(du, c.S, c.rate)
    return nothing
end

rhs!(c::Chem, du, u, k) = c(du, u, k, 0.0)

function jac!(c::Chem, J::SparseMatrixCSC{Float64}, u, k)
    nz = nonzeros(J)
    fill!(nz, 0.0)
    @inbounds for e in eachindex(c.jpos)
        o = c.jother[e]
        v = c.jnu[e] * k[c.jrxn[e]]
        if o != 0
            v *= u[o]
        end
        nz[c.jpos[e]] += v
    end
    return nothing
end

struct ChemJac
    c::Chem
end
(cj::ChemJac)(J, u, k, t) = jac!(cj.c, J, u, k)

zero_tgrad(dT, u, p, t) = (fill!(dT, 0.0); nothing)

# ----------------------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------------------

"""Solver name -> (algorithm constructor, family). All use KLU on the sparse pattern."""
const SOLVERS = Dict{String,Function}(
    "rodas5p"     => () -> Rodas5P(linsolve = KLUFactorization()),
    "rodas4p"     => () -> Rodas4P(linsolve = KLUFactorization()),
    "rodas4"      => () -> Rodas4(linsolve = KLUFactorization()),
    "rodas3"      => () -> Rodas3(linsolve = KLUFactorization()),
    "ros3p"       => () -> ROS3P(linsolve = KLUFactorization()),
    "ros34pw2"    => () -> ROS34PW2(linsolve = KLUFactorization()),
    "rosenbrock23"=> () -> Rosenbrock23(linsolve = KLUFactorization()),
    "fbdf"        => () -> FBDF(linsolve = KLUFactorization()),
    "qndf"        => () -> QNDF(linsolve = KLUFactorization()),
    "cvode"       => () -> CVODE_BDF(linear_solver = :KLU),    # KLU ordering set to AMD below
    "cvode_colamd"=> () -> CVODE_BDF(linear_solver = :KLU),    # Sundials' default ordering (COLAMD)
)
iscvode(name) = startswith(name, "cvode")
solver_names() = sort!(collect(keys(SOLVERS)))

mutable struct IntervalSolver
    name::String
    integ::Any
    k::Vector{Float64}          # the parameter vector seen by f and jac (aliased)
    carry_dt::Bool
    stats::Dict{String,Int}
end

const STAT_FIELDS = (:naccept, :nreject, :nf, :njacs, :nw, :nsolve, :nnonliniter, :nnonlinconvfail)

function make_solver(name::AbstractString, chem::Chem, P::Problem, dt::Real;
                     rtol = P.rtol, atol = P.atol, carry_dt::Bool = false)
    k = zeros(chem.nr)
    f = ODEFunction{true,SciMLBase.FullSpecialize}(chem; jac = ChemJac(chem),
                                                   jac_prototype = copy(chem.Jproto),
                                                   tgrad = zero_tgrad)
    prob = ODEProblem(f, copy(P.y0), (0.0, Float64(dt)), k)
    alg = SOLVERS[name]()
    integ = if iscvode(name)
        it = init(prob, alg; reltol = rtol, abstol = atol, save_everystep = false,
                  save_start = false, save_end = false, maxiters = 1_000_000)
        # SUNDIALS' KLU defaults to COLAMD ordering, which on these Jacobians makes every
        # factorisation about 10 times more expensive than with AMD (KLU's own default and
        # what LinearSolve uses). The ordering is applied at the next symbolic analysis,
        # i.e. at the first step after every reinit!.
        if name == "cvode"
            r = Sundials.SUNLinSol_KLUSetOrdering(it.LS.ptr, Cint(0))
            r == 0 || error("SUNLinSol_KLUSetOrdering failed ($r)")
        end
        it
    else
        init(prob, alg; reltol = rtol, abstol = atol, save_everystep = false,
             save_start = false, save_end = false, dense = false, maxiters = 1_000_000)
    end
    integ.p === k || error("integrator does not alias the parameter vector")
    IntervalSolver(String(name), integ, k,
                   carry_dt, Dict(String(s) => 0 for s in STAT_FIELDS))
end

statvals(st) = Dict(String(s) => Int(getfield(st, s)) for s in STAT_FIELDS if hasfield(typeof(st), s))

"""Solve one interval from y with the current `S.k`; returns the end state (a view of u)."""
function solve_interval!(S::IntervalSolver, y::Vector{Float64}, dt::Float64)
    integ = S.integ
    if iscvode(S.name)
        reinit!(integ, y)
    else
        # carry_dt: start the interval with the step size the controller proposed at the end
        # of the previous one (the last step itself was shortened to land on the interval
        # end). Otherwise the initial step is chosen afresh, as for a new problem.
        carry = S.carry_dt && integ.dtpropose > 0
        carry && (integ.dt = integ.dtpropose)
        reinit!(integ, y; t0 = 0.0, tf = dt, reset_dt = !carry)
    end
    cv = iscvode(S.name)
    s0 = cv ? Dict{String,Int}() : statvals(integ.stats)
    solve!(integ)
    rc = integ.sol.retcode
    SciMLBase.successful_retcode(rc) || error("$(S.name) failed with retcode $rc at t = $(integ.t)")
    abs(integ.t - dt) <= 1e-9 * dt || error("$(S.name) stopped at t = $(integ.t), not $dt")
    # OrdinaryDiffEq counters run on across reinit!; Sundials fills sol.stats from the CVODE
    # counters, which CVodeReInit restarts, so those are already per interval.
    s1 = statvals(cv ? integ.sol.stats : integ.stats)
    for (kk, v) in s1
        S.stats[kk] += v - get(s0, kk, 0)
    end
    return integ.u
end

# ----------------------------------------------------------------------------------------
# 24 h interval loop
# ----------------------------------------------------------------------------------------

struct RunResult
    key::Matrix{Float64}        # n_intervals x n_key, interval-end outputs (clipped)
    hourly::Matrix{Float64}     # 24 x n
    ro2_start::Vector{Float64}
    wall_s::Float64
    first_interval_s::Float64
    intervals_done::Int
    capped::Bool
    stats::Dict{String,Int}
end

function run24(S::IntervalSolver, P::Problem, coupling::Int; cap_s::Float64 = Inf)
    jt = P.jtab[coupling]
    nint = n_intervals(P, coupling)
    size(jt, 2) == nint || error("photolysis table length")
    dt = Float64(coupling)
    per_hour = round(Int, 3600 / coupling)
    n = P.mech.n
    y = copy(P.y0)
    key = zeros(nint, length(P.key_index))
    hourly = zeros(nint ÷ per_hour, n)
    ro2s = zeros(nint)
    for kk in keys(S.stats)
        S.stats[kk] = 0
    end
    t0 = time_ns()
    t_first = NaN
    done = 0
    capped = false
    for i in 1:nint
        ro2s[i] = rate_constants!(S.k, P, y, view(jt, :, i))
        u = solve_interval!(S, y, dt)
        @inbounds for s in 1:n
            y[s] = max(u[s], 0.0)
        end
        @inbounds for (q, s) in enumerate(P.key_index)
            key[i, q] = y[s]
        end
        if i % per_hour == 0
            hourly[i ÷ per_hour, :] .= y
        end
        for s in P.reset_zero
            y[s] = 0.0
        end
        for s in P.reset_initial
            y[s] = P.y0[s]
        end
        done = i
        i == 1 && (t_first = (time_ns() - t0) / 1e9)
        if (time_ns() - t0) / 1e9 > cap_s && i < nint
            capped = true
            break
        end
    end
    wall = (time_ns() - t0) / 1e9
    RunResult(key, hourly, ro2s, wall, t_first, done, capped, copy(S.stats))
end

# ----------------------------------------------------------------------------------------
# Accuracy metric (BENCHMARK.md section 6) and output files
# ----------------------------------------------------------------------------------------

const METRIC_FLOOR = 1e6

function load_reference(problem::AbstractString, coupling::Int;
                        dir::AbstractString = joinpath(bench_dir(), "reference"))
    JSON3.read(read(joinpath(dir, "ref_$(problem)_dt$(coupling).json"), String))
end

"""E_key and per-species maxima of a run against a reference (key species, interval ends)."""
function key_metric(R::RunResult, P::Problem, ref; upto::Int = R.intervals_done)
    per = Dict{String,Float64}()
    allv = Float64[]
    for (q, s) in enumerate(P.key_species)
        r = Float64.(collect(ref.key_species[Symbol(s)]))[1:upto]
        e = abs.(R.key[1:upto, q] .- r) ./ max.(abs.(r), METRIC_FLOOR)
        per[s] = maximum(e)
        append!(allv, e)
    end
    worst = argmax(per)
    (max = maximum(allv), median = median(allv), p95 = quantile(allv, 0.95),
     worst_species = worst, per_species = per)
end

function all_species_metric(R::RunResult, problem::AbstractString, coupling::Int;
                            dir::AbstractString = joinpath(bench_dir(), "reference"))
    yref = npzread(joinpath(dir, "ref_$(problem)_dt$(coupling)_hourly.npz"), ["y"])["y"]
    size(yref) == size(R.hourly) || return NaN
    maximum(abs.(R.hourly .- yref) ./ max.(abs.(yref), METRIC_FLOOR))
end

"""Write a run in the reference format (JSON plus `_hourly.npz`), scoreable by compare.py."""
function write_run(base::AbstractString, R::RunResult, P::Problem, coupling::Int, solver::String,
                   rtol, atol; extra = Dict{String,Any}())
    nint = size(R.key, 1)
    per_hour = round(Int, 3600 / coupling)
    meta = Dict{String,Any}(
        "format" => "tomas-gas-bench/reference/1",
        "problem" => P.name, "coupling_s" => coupling,
        "solver" => solver, "rtol" => rtol, "atol" => atol,
        "t_end_s" => [i * coupling for i in 1:nint],
        "key_species" => Dict(s => R.key[:, q] for (q, s) in enumerate(P.key_species)),
        "ro2_start" => R.ro2_start,
        "hourly_t_s" => [h * per_hour * coupling for h in 1:size(R.hourly, 1)],
        "stats" => merge(Dict{String,Any}(R.stats), extra),
    )
    open(base * ".json", "w") do io
        JSON3.write(io, meta)
    end
    npzwrite(base * "_hourly.npz", Dict("t_s" => Float64.(meta["hourly_t_s"]), "y" => R.hourly))
    return nothing
end

# ----------------------------------------------------------------------------------------
# Precompile workload: a 4-species toy problem through every solver path. The types are
# the same as for the benchmark mechanisms, so the compiled code is reused at any size.
# Disable with the PrecompileTools preference `precompile_workload = false`.
# ----------------------------------------------------------------------------------------

function toy_problem()
    # A -> B (k1), B + B -> C (k2), B + C -> A (k3, RO2-dependent), counter D
    S = sparse([1, 2, 2, 3, 2, 3, 1, 4], [1, 1, 2, 2, 3, 3, 3, 3],
               [-1.0, 1.0, -2.0, 1.0, -1.0, -1.0, 1.0, 1.0], 4, 3)
    m = Mechanism("toy", ["A", "B", "C", "D"], 4, 3, Int32[1, 2, 2], Int32[0, 2, 3], S, [2])
    jt = Dict(900 => fill(1e-3, 1, 2), 60 => fill(1e-3, 1, 30))
    Problem("toy", m, [1e-3, 1e-12, 1e-20], BitVector([false, false, true]),
            sparse([1], [1], [1.0], 3, 1), [(2, 1e-12)], jt, [1e10, 0, 0, 0], [4], [1],
            ["A", "B"], [1, 2], 1e-3, 1e4, 1800.0)
end

@setup_workload begin
    P = toy_problem()
    chem = Chem(P.mech)
    @compile_workload begin
        for name in ("rodas5p", "fbdf", "qndf", "cvode")
            S = make_solver(name, chem, P, 900.0)
            run24(S, P, 900)
        end
    end
end

end # module
