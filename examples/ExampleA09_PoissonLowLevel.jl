#= 

# A09 : Poisson-Problem with low level structures
([source code](SOURCE_URL))

This example computes the solution ``u`` of the Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with some right-hand side ``f`` on the unit cube domain ``\Omega`` on a given grid.

Here, the whole problem is assembled with low level structures

=#

module ExampleA09_PoissonProblemLowLevel2D

using GradientRobustMultiPhysics
using ExtendableGrids
using ExtendableSparse
using GridVisualize

## everything is wrapped in a main function
function main(; verbosity = 0, μ = 1, order = 2, nrefinements = 3, Plotter = nothing)

    ## set log level
    set_verbosity(verbosity)

    ## build/load any grid (here: a uniform-refined 2D unit square into triangles)
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefinements)

    ## create empty PDE description
    Problem = PDEDescription("Poisson problem")

    ## init FE space
    FEType = H1Pk{1,2,order}
    FES = FESpace{FEType}(xgrid)

    ## init matrix and right-hand side vector
    A = FEMatrix{Float64}("A", FES, FES)
    b = FEVector("b", FES)
    Solution = FEVector("u_h",FES)

    ## PREPARE ASSEMBLY LOOP

    ## quadrature formula
    qf = QuadratureRule{Float64,Triangle2D}(2*(get_polynomialorder(FEType, Triangle2D)-1))
    weights::Vector{Float64} = qf.w
    nweights::Int = length(weights)

    ## dofmap
    CellDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.CellDofs]
    ndofs4cell::Int = get_ndofs(ON_CELLS, FEType, Triangle2D)
    dof_j::Int, dof_k::Int = 0, 0

    ## FE basis evaluator
    FEBasis_id::FEBasisEvaluator{Float64} = FEBasisEvaluator{Float64,Triangle2D,Identity,ON_CELLS}(FES, qf)
    FEBasis_∇::FEBasisEvaluator{Float64} = FEBasisEvaluator{Float64,Triangle2D,Gradient,ON_CELLS}(FES, qf)
    ∇vals::Array{Float64,3} = FEBasis_∇.cvals
    idvals::Array{Float64,3} = FEBasis_id.cvals

    ## ASSEMBLY LOOP
    ncells::Int = num_cells(xgrid)
    @time for cell = 1 : ncells

        ## update FE basis evaluators
        update_febe!(FEBasis_id, cell) # 1 alloc if ncells > 512
        update_febe!(FEBasis_∇, cell) # 1 alloc if ncells > 512

        for j = 1 : ndofs4cell
            dof_j = CellDofs[j, cell]

            ## (∇v_j, ∇v_k)   
            for k = j : ndofs4cell
                dof_k = CellDofs[k, cell]
                temp = 0
                for qp = 1 : nweights
                    temp += weights[qp] * dot(view(∇vals,:,j,qp), view(∇vals,:,k,qp))
                end
                rawupdateindex!(A.entries, +, temp, dof_j, dof_k)
                if k > j
                    rawupdateindex!(A.entries, +, temp, dof_k, dof_j)
                end
            end

            ## (f, v_j) where f = 1
            for qp = 1 : nweights
                b.entries[dof_j] = weights[qp] * idvals[1, j, qp]
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
    Solution.entries .= A.entries \ b.entries
    
    ## plot solution (for e.g. Plotter = PyPlot)
    p = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, resolution = (1000,500))
    scalarplot!(p[1,1], xgrid, view(nodevalues(Solution[1]),1,:), levels = 7, title = "u_h")
    scalarplot!(p[1,2], xgrid, view(nodevalues(Solution[1], Gradient; abs = true),1,:), vscale = 0.8, levels = 0, colorbarticks = 9, title = "∇u_h (abs + quiver)")
    vectorplot!(p[1,2], xgrid, evaluate(PointEvaluator(Solution[1], Gradient)), spacing = 0.1, clear = false)
end

end