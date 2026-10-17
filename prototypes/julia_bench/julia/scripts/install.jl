using Pkg
Pkg.add(["OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore",
         "LinearSolve", "Sundials", "SciMLBase", "JSON3", "NPZ", "PrecompileTools",
         "ADTypes"])
Pkg.add(["SparseArrays", "LinearAlgebra", "Statistics", "Printf", "Dates"])
Pkg.status()
