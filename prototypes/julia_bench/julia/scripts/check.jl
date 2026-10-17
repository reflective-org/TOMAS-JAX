# Unit-test vectors of the benchmark (BENCHMARK.md section 4.1): f(y_test) and J(y_test) v,
# k at y_test and RO2(y_test), for every problem. Exit code 1 on any mismatch.
#
#     julia --project=. scripts/check.jl [problem ...]
using GasBench, JSON3, SparseArrays, LinearAlgebra, Printf

const PROBLEMS = isempty(ARGS) ?
    ["lim726_L2", "lim726_L3", "syn1353_L3", "syn4488_L3", "syn5742_L3"] : ARGS

relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

ok = true
for name in PROBLEMS
    P = load_problem(name)
    c = JSON3.read(read(joinpath(GasBench.bench_dir(), "problems", "checks_$(name).json"), String))
    y = Float64.(collect(c.y_test)); v = Float64.(collect(c.v))
    chem = Chem(P.mech)
    k = zeros(P.mech.nr)
    ro2 = GasBench.rate_constants!(k, P, y, view(P.jtab[900], :, 1))
    kref = Float64.(collect(c.k_test))
    ek = maximum(abs.(k .- kref) ./ max.(abs.(kref), floatmin()))
    du = zeros(P.mech.n)
    rhs!(chem, du, y, kref)
    fref = Float64.(collect(c.f_test))
    # per-species error relative to the scale of the terms that are summed
    ef = relerr(du, fref)
    J = copy(chem.Jproto)
    jac!(chem, J, y, kref)
    jv = J * v
    jvref = Float64.(collect(c.jac_v_test))
    ej = relerr(jv, jvref)
    # Jacobian against a central finite difference along v (a sanity check of the analytic J)
    h = 1e-2   # f is at most quadratic in y, so the central difference is exact up to rounding
    fp = zeros(P.mech.n); fm = zeros(P.mech.n)
    rhs!(chem, fp, y .+ h .* v, kref); rhs!(chem, fm, y .- h .* v, kref)
    efd = relerr((fp .- fm) ./ (2h), jv)
    nnz_ok = GasBench.jac_nnz(chem) == Int(c.jac_nnz)
    pass = ek < 1e-14 && abs(ro2 - c.ro2_test) / c.ro2_test < 1e-14 && ef < 1e-13 && ej < 1e-13 && nnz_ok
    global ok &= pass
    @printf("%-12s n=%5d nr=%6d nnz=%6d (ref %6d, pattern with diagonal %6d)  k %.1e  RO2 %.1e  f %.1e  Jv %.1e  FD-check %.1e  %s\n",
            name, P.mech.n, P.mech.nr, GasBench.jac_nnz(chem), Int(c.jac_nnz), nnz(chem.Jproto), ek,
            abs(ro2 - c.ro2_test) / c.ro2_test, ef, ej, efd, pass ? "PASS" : "FAIL")
end
exit(ok ? 0 : 1)
