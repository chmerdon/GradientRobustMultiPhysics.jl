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
    estimator4face .*= FE.grid.length4faces[:].^(3 - 2*derivative)
    total_estimator = sqrt.(sum(estimator4face, dims = 1))[1]

    ncells = size(FE.grid.nodes4cells,1)
    FESolveCommon.assemble_operator!(estimator4cell,FESolveCommon.CELL_L2_FplusLA,FE,PD.volumedata4region[1],PD.quadorder4region[1],discrete_solution,PD.diffusion) 
    estimator4cell .*= FE.grid.volume4cells[:].^(2 - derivative) # todo: use diameter
    total_estimator += sqrt.(sum(estimator4cell, dims = 1))[1]

    estimator4cell += sum(estimator4face[FE.grid.faces4cells], dims = 2)


    return total_estimator
end

function estimate_cellwise!(estimator4cell::Array, PD::FESolvePoisson.PoissonProblemDescription, FE_stress::AbstractFiniteElement, FE_divergence::AbstractFiniteElement, discrete_solution::Array)

    # tangent jumps (full jumps are computed, but normal-jumps are zero)
    nfaces = size(FE_stress.grid.nodes4faces,1)
    estimator4face = zeros(Float64,nfaces,2)
    FESolveCommon.assemble_operator!(estimator4face,FESolveCommon.FACE_L2_JumpA,FE_stress,discrete_solution)
    estimator4face .*= FE_stress.grid.length4faces[:]
    total_estimator = sqrt.(sum(sum(estimator4face, dims=2), dims = 1))[1]

    # volume term h_T || f + div(sigma) ||
    ncells = size(FE_stress.grid.nodes4cells,1)
    FESolveCommon.assemble_operator!(estimator4cell,FESolveCommon.CELL_L2_FplusDIVA,FE_stress,PD.volumedata4region[1],PD.quadorder4region[1],discrete_solution)
    estimator4cell .*= FE_stress.grid.volume4cells[:] # todo: use diameter
    total_estimator += sqrt.(sum(estimator4cell, dims = 1))[1]
    estimator4cell[:] += sum(estimator4face[FE_stress.grid.faces4cells], dims = 2)

    # 2nd volume term h_T || curl(sigma) || (vanishes for RT0)
    estimator4cell2 = zeros(Float64,ncells)
    FESolveCommon.assemble_operator!(estimator4cell2,FESolveCommon.CELL_L2_CURLA,FE_stress,discrete_solution)
    estimator4cell2 .*= FE_stress.grid.volume4cells[:] # todo: use diameter
    total_estimator += sqrt.(sum(estimator4cell2, dims = 1))[1]
    estimator4cell += estimator4cell2


    return total_estimator
end

end
