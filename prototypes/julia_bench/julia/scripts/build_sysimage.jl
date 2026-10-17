# Build a PackageCompiler system image with GasBench and its solver stack.
#   julia --project=sysimage/tools scripts/build_sysimage.jl
using PackageCompiler
here = normpath(joinpath(@__DIR__, ".."))
t0 = time()
create_sysimage(["GasBench", "JSON3", "NPZ"];
                sysimage_path = joinpath(here, "sysimage", "gasbench.dylib"),
                project = joinpath(here, "sysimage", "env"),
                precompile_execution_file = joinpath(here, "scripts", "precompile_exec.jl"))
println("sysimage built in ", round(time() - t0; digits = 1), " s")
