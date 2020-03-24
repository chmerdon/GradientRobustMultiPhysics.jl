module FEEstimate

export estimate_cellwise!

using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using FESolvePoisson
using Grid

# compute error estimator for Poisson problem solutions
# derivative = 0 gives estimator for L2 norm
# derivative = 1 gives estimator for H1 semi-norm
function estimate_cellwise!(estimator4cell::Array, PD::FESolvePoisson.PoissonProblemDescription, FE::AbstractFiniteElement, discrete_solution::Array, derivative::Int = 1)
    nfaces = size(FE.grid.nodes4faces,1)
    estimator4face = zeros(Float64,nfaces,2)
    FESolveCommon.assemble_operator!(estimator4face,FESolveCommon.FACE_L2_JumpDA,FE,discrete_solution)
    total_estimator = sqrt.(sum(sum(estimator4face, dims=2).*FE.grid.length4faces[:].^(3 - 2*derivative), dims = 1))[1]

    ncells = size(FE.grid.nodes4cells,1)
    FESolveCommon.assemble_operator!(estimator4cell,FESolveCommon.CELL_L2_FplusLA,FE,PD.volumedata4region[1],PD.quadorder4region[1],discrete_solution,PD.diffusion)
    total_estimator += sqrt.(sum(estimator4cell.*FE.grid.volume4cells[:].^(4 - 2*derivative), dims = 1))[1]

    estimator4cell += sum(estimator4face[FE.grid.faces4cells], dims = 2)


    return total_estimator
end

end
