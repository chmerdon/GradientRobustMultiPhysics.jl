#= 

# A09 : Poisson-Problem with low level structures
([source code](SOURCE_URL))

This example computes the solution ``u`` of the Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f = 1`` on the unit square domain ``\Omega`` on a given grid.

Here, the whole problem is assembled with low level structures

=#

module ExampleA09_PoissonLowLevel

using GradientRobustMultiPhysics
using ExtendableGrids
using ExtendableSparse
using GridVisualize

const f = x -> 1

## everything is wrapped in a main function
function main(; verbosity = 0, μ = 1, nrefinements = 3, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## build/load any grid (here: a uniform-refined 2D unit square into triangles)
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefinements)

    ## choose FE type
    FEType = H1P2{1,2}

    ## ASSEMBLY

    ## init FEspace, matrix and right-hand side vector
    FES = FESpace{FEType}(xgrid)
    A = FEMatrix{Float64}("A", FES, FES)
    b = FEVector("b", FES)

    ## quadrature formula
    qf = QuadratureRule{Float64,Triangle2D}(2*(get_polynomialorder(FEType, Triangle2D)-1))
    weights::Vector{Float64} = qf.w
    nweights::Int = length(weights)

    ## dofmap
    CellDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.CellDofs]
    ndofs4cell::Int = get_ndofs(ON_CELLS, FEType, Triangle2D)
    dof_j::Int, dof_k::Int = 0, 0

    ## FE basis evaluator
    FEBasis_id::FEEvaluator{Float64} = FEEvaluator(FES, Identity, qf)
    FEBasis_∇::FEEvaluator{Float64} = FEEvaluator(FES, Gradient, qf)
    idvals::Array{Float64,3} = FEBasis_id.cvals
    ∇vals::Array{Float64,3} = FEBasis_∇.cvals
    L2G::L2GTransformer{Float64, Int32, Triangle2D} = FEBasis_∇.L2G
    L2GAinv::Matrix{Float64} = zeros(Float64,2,2)

    ## ASSEMBLY LOOP
    ncells::Int = num_cells(xgrid)
    x::Vector{Float64} = zeros(Float64, 2)
    cellvolumes = xgrid[CellVolumes]
    @time for cell::Int = 1 : ncells

        ## update FE basis evaluators
        FEBasis_∇.citem[] = cell
        update_basis!(FEBasis_∇) 

        for j = 1 : ndofs4cell
            dof_j = CellDofs[j, cell]

            ## (∇v_j, ∇v_k)   
            for k = j : ndofs4cell
                dof_k = CellDofs[k, cell]
                temp = 0
                for qp = 1 : nweights
                    temp += weights[qp] * μ * dot(view(∇vals,:,j,qp), view(∇vals,:,k,qp)) * cellvolumes[cell]
                end
                rawupdateindex!(A.entries, +, temp, dof_j, dof_k)
                if k > j
                    rawupdateindex!(A.entries, +, temp, dof_k, dof_j)
                end
            end

            for qp = 1 : nweights
                ## get global x for quadrature point
                update_trafo!(L2G, cell)
                eval_trafo!(x, L2G, FEBasis_∇.xref[qp])

                ## (f, v_j)
                b.entries[dof_j] += weights[qp] * idvals[1, j, qp] * f(x) * cellvolumes[cell]
            end
        end
    end

    ## fix boundary dofs
    BFaceDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.BFaceDofs]
    ndofs4bface::Int = get_ndofs(ON_BFACES, FEType, Edge1D)
    nbfaces::Int = num_sources(BFaceDofs)
    for bface = 1 : nbfaces
        for j = 1 : ndofs4bface
            dof_j = BFaceDofs[j, bface]
            A.entries[dof_j,dof_j] = 1e60
            b.entries[dof_j] = 0
        end
    end

    ## solve
    ExtendableSparse.flush!(A.entries)
    Solution = FEVector("u_h", FES)
    Solution.entries .= A.entries \ b.entries
    
    ## plot solution (for e.g. Plotter = PyPlot)
    p = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, resolution = (1000,500))
    scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]),1,:), levels = 7, title = "u_h")
    scalarplot!(p[1,2], xgrid, view(nodevalues(Solution[1], Gradient; abs = true),1,:), vscale = 0.8, levels = 0, colorbarticks = 9, title = "∇u_h (abs + quiver)")
    vectorplot!(p[1,2], xgrid, evaluate(PointEvaluator(Solution[1], Gradient)), spacing = 0.1, clear = false)
end

end