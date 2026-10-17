# Workload run while PackageCompiler builds the system image: the real lim726 problem through
# every solver path for a few hours of simulated time, plus the scoring and output code.
using GasBench, JSON3, NPZ
P = load_problem("lim726_L3")
chem = Chem(P.mech)
ref = GasBench.load_reference("lim726_L3", 900)
for name in ("rodas5p", "fbdf", "qndf", "cvode", "rodas4p", "ros3p")
    S = make_solver(name, chem, P, 900)
    R = run24(S, P, 900; cap_s = 0.5)
    GasBench.key_metric(R, P, ref)
end
S = make_solver("rodas5p", chem, P, 60)
R = run24(S, P, 60; cap_s = 0.5)
tmp = mktempdir()
R2 = run24(make_solver("cvode", chem, P, 900), P, 900)
GasBench.write_run(joinpath(tmp, "x"), R2, P, 900, "x", 1e-3, 1e4)
GasBench.all_species_metric(R2, "lim726_L3", 900)
JSON3.write(Dict("a" => 1.0))
