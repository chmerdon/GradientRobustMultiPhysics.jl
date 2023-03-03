### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 014c3380-b361-11ed-273e-37541f1ed74f
begin
	using GradientRobustMultiPhysics
	using ExtendableGrids
	using ExtendableSparse
	using GridVisualize
	using PlutoVista
	using Pkg

	## make sure to use the newest version
	#Pkg.add(Pkg.PackageSpec(;name="GradientRobustMultiPhysics", version="0.11.1"))
	Pkg.status()
end

# ╔═╡ 3e8fd4f5-b6be-4019-9c76-a80cc985b70e
md"""
# Tutorial notebook: Low level structures and Poisson problem

As a toy example in this notebook, consider the Poisson problem that seeks ``u`` such that
```math
\begin{aligned}
	- \mu \Delta u = f.
\end{aligned}
```

The weak formulation seeks ``u \in V := H^1_0(\Omega)`` such that
```math
\begin{aligned}
	\mu (\nabla u, \nabla v) = (f, v)
	\quad \text{for all } v \in V
\end{aligned}
```

"""

# ╔═╡ 6160364e-6ab2-475d-9345-3436e6f2b3e3
begin
	## PDE data
	μ = 1.0
	f = x -> x[1] - x[2]
	quadorder_f = 1

	## discretization parameters
	nref = 9
	order = 1
end

# ╔═╡ 8e0a11e7-7998-403a-8484-7d0180ea40b2
begin
	## create grid
	X = LinRange(0,1,2^nref+1)
	Y = LinRange(0,1,2^nref+1)
	println("Creating grid...")
	@time xgrid = simplexgrid(X,Y)
	println("Preparing FaceNodes...")
	@time xgrid[FaceNodes]
	println("Preparing CellVolumes...")
    @time xgrid[CellVolumes]
	xgrid
end

# ╔═╡ 7e2a082f-5457-47ac-96d5-b688a70a5506
begin
	## create finite element space
    FEType = H1Pk{1,2,order}

	## prepare finite element space and dofmaps
	println("Creating FESpace...")
    @time FES = FESpace{FEType}(xgrid)
	println("Creating cell dofs...")
    @time CellDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.CellDofs]
	println("Creating bface dofs...")
    @time BFaceDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.BFaceDofs]
	
	FES
end

# ╔═╡ 6dcca265-b8fd-4043-a2a7-4ff9bf579dd5
function assemble_Laplacian!(A::ExtendableSparseMatrix, FES, μ = 1)

    xgrid = FES.xgrid
    EG = xgrid[UniqueCellGeometries][1]
    FEType = eltype(FES)

    ## quadrature formula
    qf = QuadratureRule{Float64, EG}(2*(get_polynomialorder(FEType, EG)-1))
    weights::Vector{Float64} = qf.w
    nweights::Int = length(weights)

    ## dofmap
    CellDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.CellDofs]
    ndofs4cell::Int = get_ndofs(ON_CELLS, FEType, EG)
    dof_j::Int, dof_k::Int = 0, 0

    ## FE basis evaluator
    FEBasis_∇::FEEvaluator{Float64} = FEEvaluator(FES, Gradient, qf)
    ∇vals::Array{Float64,3} = FEBasis_∇.cvals
   
    ## local matrix and vector structures
    Aloc = zeros(Float64, ndofs4cell, ndofs4cell)

    ## ASSEMBLY LOOP
    ncells::Int = num_cells(xgrid)
    cellvolumes = xgrid[CellVolumes]

    for cell = 1 : ncells

        ## update FE basis evaluators
        FEBasis_∇.citem[] = cell
        update_basis!(FEBasis_∇) 

		## assemble local stiffness matrix
        for j = 1 : ndofs4cell, k = j : ndofs4cell
			temp = 0
			for qp = 1 : nweights
				temp += weights[qp] * μ * dot(view(∇vals,:,j,qp), view(∇vals,:,k,qp))
			end
			Aloc[j,k] = temp
        end
        Aloc .*= cellvolumes[cell]

		## add local matrix to global matrix
        for j = 1 : ndofs4cell
            dof_j = CellDofs[j, cell]
            for k = j : ndofs4cell
                dof_k = CellDofs[k, cell]
                if abs(Aloc[j,k]) > 1e-15
                    # write into sparse matrix, only lines with allocations
                    rawupdateindex!(A, +, Aloc[j,k], dof_j, dof_k) 
                    if k > j
                        rawupdateindex!(A, +, Aloc[j,k], dof_k, dof_j)
                    end
                end
            end
        end

        fill!(Aloc, 0)
    end
    flush!(A)
end

# ╔═╡ 783d15ab-5205-43d6-9fe6-b0d44a781c92
function assemble_rhs!(b::Vector, FES, f)

    if f === nothing
        fill!(b, 0)
        return 
    end

    xgrid = FES.xgrid
    EG = xgrid[UniqueCellGeometries][1]
    FEType = eltype(FES)

    ## quadrature formula
    qf = QuadratureRule{Float64, EG}(get_polynomialorder(FEType, EG) + quadorder_f)
    weights::Vector{Float64} = qf.w
    xref = qf.xref
    nweights::Int = length(weights)

    ## dofmap
    CellDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.CellDofs]
    ndofs4cell::Int = get_ndofs(ON_CELLS, FEType, EG)

    ## FE basis evaluator
    FEBasis_id::FEEvaluator{Float64} = FEEvaluator(FES, Identity, qf)
    idvals::Array{Float64,3} = FEBasis_id.cvals
    L2G::L2GTransformer{Float64, Int32, EG} = L2GTransformer(EG, xgrid, ON_CELLS)

    ## ASSEMBLY LOOP
    function barrier(L2G::L2GTransformer{Tv,Ti,Tg,Tc}) where {Tv,Ti,Tg,Tc}
        
        bloc = zeros(Float64, ndofs4cell)
        ncells::Int = num_cells(xgrid)
        dof_j::Int = 0
        x::Vector{Float64} = zeros(Float64, 2)
        cellvolumes = xgrid[CellVolumes]
        
        for cell = 1 : ncells
            for j = 1 : ndofs4cell
                ## right-hand side
                temp = 0
                for qp = 1 : nweights
                    ## get global x for quadrature point
                    update_trafo!(L2G, cell)
                    eval_trafo!(x, L2G, xref[qp])
                    ## (f, v_j)
                    temp += weights[qp] * idvals[1, j, qp] * f(x)
                end
                bloc[j] = temp
            end
            
            for j = 1 : ndofs4cell
                dof_j = CellDofs[j, cell]
                b[dof_j] += bloc[j] * cellvolumes[cell]
            end
            
            fill!(bloc, 0)
        end
    end
    barrier(L2G)
end

# ╔═╡ 7299586b-6859-41af-bcd0-1a0daa454c81
function solve_poisson_lowlevel(FES, μ, f)
	
	Solution = FEVector(FES)
	FES = Solution[1].FES
	A = FEMatrix(FES, FES)
	b = FEVector(FES)
	println("Assembling Laplacian...")
	@time assemble_Laplacian!(A.entries, FES, μ)
	println("Assembling right-hand side...")
	@time assemble_rhs!(b.entries, FES, f)

    ## fix boundary dofs
	println("Assembling boundary data...")
	@time begin
		BFaceDofs::Adjacency{Int32} = FES[GradientRobustMultiPhysics.BFaceDofs]
		nbfaces::Int = num_sources(BFaceDofs)
		AM::ExtendableSparseMatrix{Float64,Int64} = A.entries
		dof_j::Int = 0
		for bface = 1 : nbfaces
			for j = 1 : num_targets(BFaceDofs,1)
				dof_j = BFaceDofs[j, bface]
				AM[dof_j,dof_j] = 1e60
				b.entries[dof_j] = 0
			end
		end
	end
	ExtendableSparse.flush!(A.entries)

    ## solve
	println("Solving linear system...")
    @time copyto!(Solution.entries, A.entries \ b.entries)

	return Solution
end

# ╔═╡ 0785624c-e95a-4e76-8071-2eea591091e0
begin
	## call low level solver
	sol = solve_poisson_lowlevel(FES, μ, f)
end

# ╔═╡ 6f7a1407-dfb3-497b-8c57-8efac6592194
tricontour(xgrid[Coordinates],xgrid[CellNodes],sol.entries[1:num_nodes(xgrid)]; levels = 5)

# ╔═╡ 35386942-6556-45df-8be3-10a02b05e854
md"""
	Now everything again high-level via a PDEDescription:
"""

# ╔═╡ 02740215-fde2-414c-8cdf-554aaa6c1367
begin
    ## create PDE description
	Problem = PDEDescription("Poisson problem")
	add_unknown!(Problem; unknown_name = "u", equation_name = "Poisson equation")
	add_operator!(Problem, [1,1], LaplaceOperator(μ))
	fdata = DataFunction((result, x) -> (result[1] = f(x);), [1,2]; dependencies = "X", bonus_quadorder = quadorder_f)
	add_rhsdata!(Problem, 1, LinearForm(Identity, fdata; regions = [1]))
	add_boundarydata!(Problem, 1, [1,2,3,4], HomogeneousDirichletBoundary)
	Problem
end

# ╔═╡ 07a09bf6-b821-40ce-8056-19dbf156b223
begin
    ## solve
    @time sol_high = solve(Problem, FES; show_statistics = true)
end

# ╔═╡ Cell order:
# ╠═014c3380-b361-11ed-273e-37541f1ed74f
# ╟─3e8fd4f5-b6be-4019-9c76-a80cc985b70e
# ╠═6160364e-6ab2-475d-9345-3436e6f2b3e3
# ╟─6f7a1407-dfb3-497b-8c57-8efac6592194
# ╠═8e0a11e7-7998-403a-8484-7d0180ea40b2
# ╠═7e2a082f-5457-47ac-96d5-b688a70a5506
# ╠═0785624c-e95a-4e76-8071-2eea591091e0
# ╠═7299586b-6859-41af-bcd0-1a0daa454c81
# ╠═6dcca265-b8fd-4043-a2a7-4ff9bf579dd5
# ╠═783d15ab-5205-43d6-9fe6-b0d44a781c92
# ╟─35386942-6556-45df-8be3-10a02b05e854
# ╠═02740215-fde2-414c-8cdf-554aaa6c1367
# ╠═07a09bf6-b821-40ce-8056-19dbf156b223
