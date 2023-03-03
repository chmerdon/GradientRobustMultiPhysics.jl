### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ fa059506-0710-11ec-1bb4-5f0937e3b169
begin
	using GradientRobustMultiPhysics
	using ExtendableGrids
	using SimplexGridFactory
	using ForwardDiff
	using Triangulate
	using GridVisualize
	using PlutoVista
	default_plotter!(PlutoVista)
	using PlutoUI
	using Pkg
	#Pkg.add(Pkg.PackageSpec(;name="GradientRobustMultiPhysics", rev="master"))
	Pkg.status()
end

# ╔═╡ 582e3416-5d15-4d01-936a-bd2a1e6b7310
md"""
## Tutorial notebook: Cahn-Hilliard (mixed form)

Seek ``(c,\mu)`` such that
```math
\begin{aligned}
c_t - M \Delta \mu & = 0\\
\mu - \partial f / \partial c + \lambda \Delta c & = 0.
\end{aligned}
```

with ``f(c) = 100c^2(1-c)^2``, constant parameters ``M`` and ``\lambda`` and (random)
initial concentration as defined in the code below.

A weak solution is characterised by
```math
\begin{aligned}
(c_t, v) + M (\nabla \mu, \nabla v) & = 0 & \text{for all } v \in V,\\
(\mu - \partial f / \partial c, w) - \lambda (\nabla c, \nabla w) & = 0 & \text{for all } w \in V.
\end{aligned}
```


### Problem description
"""

# ╔═╡ e882b9a6-5804-4381-a09c-bf9df81d798a
begin
	## PROBLEM DATA
	f = (c) -> 100*c^2*(1-c)^2 
	dfdc = (c) -> ForwardDiff.derivative(f, c)
	M = 1.0
	λ = 1e-2
	c0 = DataFunction((result, x) -> (result[1] = 0.1 + 0.8*x[1] + 0.02 * (0.5 - rand());), [1,2]; dependencies = "X", bonus_quadorder = 2)

	## PROBLEM DEFINITION
    Problem = PDEDescription("Cahn-Hilliard")
    add_unknown!(Problem; unknown_name = "c", equation_name = "concentration eq.")
    add_unknown!(Problem; unknown_name = "μ", equation_name = "chemical pot. eq.")
    add_operator!(Problem, [1,2], LaplaceOperator(M; store = true))
    add_operator!(Problem, [2,2], ReactionOperator(1; store = true))
    add_operator!(Problem, [2,1], LaplaceOperator(-λ; store = true))

	## AD-NEWTON for nonlinear reaction part (= -df/dc times test function)
    function ∂f∂c_kernel(result, input)
		# input = [eval of Identity of unknown 1 = c_h]
        result[1] = -dfdc(input[1]) # to be multiplied with Identity of testfunction
    end
    add_operator!(Problem, 2, NonlinearForm(Identity, [Identity], [1], ∂f∂c_kernel, [1,1]; name = "(-∂f/∂c, w)", newton = true))
    
	@info Problem
end

# ╔═╡ ef7b06ec-7614-4048-a50b-90301dd24b32
md"""
### Grid generation
"""

# ╔═╡ 6243a27b-706f-4d4b-acbd-5431868e2b6f
begin
	## GRID
	nref = 6
	grid = simplexgrid(Triangulate;
                points=[0 0 ; 0 1 ; 1 1 ; 1 0]',
                bfaces=[1 2 ; 2 3 ; 3 4 ; 4 1 ]',
                bfaceregions=[1, 2, 3, 4],
                regionpoints=[0.5 0.5;]',
                regionnumbers=[1],
                regionvolumes=[4.0^-(nref)])
	@info grid
	gridplot(grid)
end

# ╔═╡ 575be54b-e46a-4576-9190-e826d37def7c
md"""
### Finite element spaces
"""

# ╔═╡ 0de84d2f-9371-4352-b4a0-9b446cb297f8
begin
	## FINITE ELEMENT SPACES
	order = 1
	FES = [FESpace{H1Pk{1,2,order}}(grid), FESpace{H1Pk{1,2,order}}(grid)]
	sol = FEVector(FES)	## show information about blocks
	@info sol
end

# ╔═╡ d1994d6f-e77f-4a7e-8daa-f60ef8df4dff
md"""
### Solving
"""

# ╔═╡ 0301f52c-7b36-46c5-aea1-958dece2b304
begin
	## INITIAL CONDITION
	interpolate!(sol[1], c0)
	
	## PREPARE TIME EVOLUTION
	tcs = TimeControlSolver(Problem, sol, BackwardEuler;
                    timedependent_equations = [1],   # only c has time derivative
                    maxiterations = 50,
					dt_lump = [2.0], # lumps and assembles dt_lump*diag(mass matrix)
                    target_residual = 1e-10)
	
	## EVOLVE
	τ = 5e-6 # time step
	nsteps = 100
	nviews_c = nodevalues_view(sol[1])
	sol4t = [nviews_c[1][:]] # save solution at all timesteps
	for j = 1 : nsteps
		advance!(tcs, τ) # single time step advance
		# print current time, iterations, nonlinear residuals for each equation
		@info tcs.ctime, Int(tcs.statistics[1,4]), tcs.statistics[:,3]
		push!(sol4t, nviews_c[1][:])
	end
end

# ╔═╡ 0eb8b37f-c08f-4fd2-aa7e-d745336f0ee0
md"""
## Results
"""

# ╔═╡ 90c83f13-d298-45c7-90d9-1739b3e810fb
md"""
Time: $(@bind t Slider(0:length(sol4t)-1,show_value=true,default=0))
"""

# ╔═╡ 136ddff3-82bc-458e-952e-8afc13982cf4
scalarplot(grid, sol4t[t+1]; limits = (-0.1,1.1), levels = 1, title = "c ( T=$(t*τ))")

# ╔═╡ Cell order:
# ╠═fa059506-0710-11ec-1bb4-5f0937e3b169
# ╟─582e3416-5d15-4d01-936a-bd2a1e6b7310
# ╠═e882b9a6-5804-4381-a09c-bf9df81d798a
# ╟─ef7b06ec-7614-4048-a50b-90301dd24b32
# ╠═6243a27b-706f-4d4b-acbd-5431868e2b6f
# ╟─575be54b-e46a-4576-9190-e826d37def7c
# ╠═0de84d2f-9371-4352-b4a0-9b446cb297f8
# ╟─d1994d6f-e77f-4a7e-8daa-f60ef8df4dff
# ╠═0301f52c-7b36-46c5-aea1-958dece2b304
# ╟─0eb8b37f-c08f-4fd2-aa7e-d745336f0ee0
# ╟─90c83f13-d298-45c7-90d9-1739b3e810fb
# ╠═136ddff3-82bc-458e-952e-8afc13982cf4
