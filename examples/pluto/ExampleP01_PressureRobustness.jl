### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ fa059506-0710-11ec-1bb4-5f0937e3b169
begin
	using GradientRobustMultiPhysics
	using PlutoVista
	using PlutoUI
end

# ╔═╡ 5fd62983-cad4-4a72-97e3-51c512451753
md"""
# ExampleP01: Pressure-Robustness

In this example the effect of pressure-robustness in the linear Stokes problem 

```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = \mathbf{f}\\
\mathrm{div}(u) & = 0
\end{aligned}
```

is demonstrated. If the right-hand side ``\mathbf{f}`` is a gradient, discretisation errors of
non pressure-robust discretisations scale with the inverse of the viscosity ``\mu``.
To test this we prescribe the boundary data of a linear velocity field ``\mathbf{u}(x,y) = [1+y,-x-1]`` and ``\mathbf{f} = \nabla(\sin(x)\cos(y))``.

###### The following intended interactivities are possible:

- In Step 3 the user can choose a finite-element pair and if a pressure-robust modification to the right-hand side should be applied.

- In Step 4 the user can specify the viscosity coefficient and see how the discretisation errors depend on it (automatically updated plot in Step 5).

###### Note: 
To run this notebook with Pluto, additional packages are needed (see below).

"""

# ╔═╡ 582e3416-5d15-4d01-936a-bd2a1e6b7310
md"""
### Step 1: Mesh definition
"""

# ╔═╡ e882b9a6-5804-4381-a09c-bf9df81d798a
begin
	xgrid = grid_unitsquare(Triangle2D) # initial grid
    xgrid = uniform_refine(xgrid,3) # uniform refinement
	xCoordinates = xgrid[Coordinates]
	nnodes = size(xCoordinates,2)
	@show "The grid has $nnodes nodes (unhide me for details)"
end

# ╔═╡ bf1bb705-bdcc-4e10-9711-747fbbebbaef
md"""
### Step 2: Stokes problem definition
"""

# ╔═╡ d44e5d2f-5483-4336-b000-1376b29e1b6f
begin
	## data
	f! = (result,x) -> (result[1] = cos(x[1])*cos(x[2]); result[2] = -sin(x[1])*sin(x[2]))
	u! = (result,x) -> (result[1] = 1+x[2]; result[2] = -x[1]-1)
	u = DataFunction(u!, [2,2]; name = "u", dependencies = "X", quadorder = 1)
	f = DataFunction(f!, [2,2]; name = "f", dependencies = "X", quadorder = 5)
	
	## problem description
    Problem = PDEDescription("Stokes problem")
    add_unknown!(Problem; equation_name = "momentum eq.", unknown_name = "u")
    add_unknown!(Problem; equation_name = "div constraint", unknown_name = "p")
    add_boundarydata!(Problem, 1, [1,2,3,4], InterpolateDirichletBoundary; data = u)
    add_constraint!(Problem, FixedIntegralMean(2,0))

    # add Laplacian for velocity
    Lop = add_operator!(Problem, [1,1], LaplaceOperator(1; store = true))

    # add Lagrange multiplier for divergence of velocity
    add_operator!(Problem, [1,2], LagrangeMultiplier(Divergence; store = true))
    
  	# add right-hand side operator
    Rop = add_rhsdata!(Problem, 1, RhsOperator(Identity, [0], f))
	
	@show "unhide me for details"
end

# ╔═╡ 5bb01852-d3f5-4512-b8b4-fb5442eaeb97
md"""
### Step 3: Choose discretisation

The following two radio buttons allow to select the finite element pair and the right-hand side discretistion classical or pressure-robust).
"""

# ╔═╡ f74bd3d6-ee33-4f77-9f3b-a53be134041d
@bind method Radio(["Bernardi-Raugel", "enriched Taylor-Hood (P2-bubble)"], default = "Bernardi-Raugel")

# ╔═╡ b5fbbd91-0484-4bd8-a34d-4bab2d0588d6
@bind reconstruct Radio(["classical", "pressure-robust"], default = "classical")

# ╔═╡ bf4a310d-b12f-4f55-a367-39b06b3ee228
begin
	if method == "Bernardi-Raugel"
    	FETypes = [H1BR{2}, H1P0{1}] # Bernardi--Raugel pair
		offset = nnodes
	elseif method == "enriched Taylor-Hood (P2-bubble)"
		FETypes = [H1P2B{2,2}, H1P1{1}] # P2-bubble pair
		offset = nnodes + num_sources(xgrid[FaceNodes]) + num_sources(xgrid[CellNodes])
	else
		@show "Please specify the method above"
	end
		
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid; broken = true)]
	Solution = FEVector{Float64}(["u_h", "p_h"],FES)
	
	## choose reconstrution operator in right-hand side
	if reconstruct == "classical"
		VIdentity = Identity # classical scheme
	elseif reconstruct == "pressure-robust"
		if FETypes[1] == H1BR{2}
    		VIdentity = ReconstructionIdentity{HDIVBDM1{2}}
		elseif FETypes[1] == H1P2B{2,2}
			VIdentity = ReconstructionIdentity{HDIVBDM2{2}}
		else
			@show "Please specify the method above"
		end
	else
		@show "Please specify the method above"
	end
	
	## overwrite the right-hand side operator with the new operator
	Problem.RHSOperators[1][Rop] = RhsOperator(VIdentity, [0], f)
	
	@show "Currently using the $reconstruct $method scheme (unhide me for details)"
end

# ╔═╡ ac3cb451-b811-4a34-958c-b8266b6c4f17
md"""
### Step 4: Choose viscosity and solve

Move the slider to change the viscosity coefficient.
"""

# ╔═╡ df2b6a48-e82b-4894-961f-8cc251eb0a01
md"""
log(μ): $(@bind logmu Slider(-8:0.1:8,show_value=true, default = -6))
"""

# ╔═╡ 84cc43bb-33bf-472e-be76-1a570f772e82
 μ = 10^Float64(logmu)

# ╔═╡ 4e5e4d65-2db6-48d2-b8ac-5b5eb5d505e7
begin
	Problem.LHSOperators[1,1][Lop].factor = μ # assign viscosity factor to Laplacian
	residual = solve!(Solution, Problem)
	
	@show "Solve finished with residual = $residual"
end

# ╔═╡ 6243a27b-706f-4d4b-acbd-5431868e2b6f
md"""
### Step 5: Plot of 1st and 2nd component of u
"""

# ╔═╡ b87e20a7-2d97-430c-b18f-da849ee8ebc9
tricontour(xgrid[Coordinates],xgrid[CellNodes],view(Solution.entries,1:nnodes);isolines=11, title = "$μ")

# ╔═╡ 038bcbb0-6cc5-4a20-908d-e2a44dd7ef9c
tricontour(xgrid[Coordinates],xgrid[CellNodes],view(Solution.entries,offset+1:offset+nnodes);isolines=11, title = "$μ")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GradientRobustMultiPhysics = "0802c0ca-1768-4022-988c-6dd5f9588a11"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PlutoVista = "646e1f28-b900-46d7-9d87-d554eb38a413"

[compat]
GradientRobustMultiPhysics = "~0.6.1"
PlutoUI = "~0.7.9"
PlutoVista = "~0.3.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "a0fcc1bb3c9ceaf07e1d0529c9806ce94be6adf9"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.9"

[[ExtendableGrids]]
deps = ["AbstractTrees", "Dates", "DocStringExtensions", "ElasticArrays", "InteractiveUtils", "LinearAlgebra", "Printf", "Random", "SparseArrays", "Test"]
git-tree-sha1 = "5a42a9371dd5ad1a00ec27c63c5eccc8e38dab43"
uuid = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
version = "0.7.9"

[[ExtendableSparse]]
deps = ["DocStringExtensions", "LinearAlgebra", "Printf", "Requires", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "0341e41e45e6c5e46be89c33543082a0a867707c"
uuid = "95c220a8-a1cf-11e9-0c77-dbfce5f500b3"
version = "0.6.5"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7c365bdef6380b29cfc5caaf99688cd7489f9b87"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.2"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "15ff9a14b9e1218958d3530cc288cf31465d9ae2"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.13"

[[GradientRobustMultiPhysics]]
deps = ["DiffResults", "DocStringExtensions", "ExtendableGrids", "ExtendableSparse", "ForwardDiff", "GridVisualize", "LinearAlgebra", "Logging", "Printf", "SparseArrays", "StaticArrays", "SuiteSparse", "Test", "WriteVTK"]
git-tree-sha1 = "6279979ab9802a0aa66b7cd5a1c78f963210e1d0"
uuid = "0802c0ca-1768-4022-988c-6dd5f9588a11"
version = "0.6.1"

[[GridVisualize]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "ElasticArrays", "ExtendableGrids", "GeometryBasics", "LinearAlgebra", "OrderedCollections", "PkgVersion", "Printf", "StaticArrays"]
git-tree-sha1 = "712d795073f49e173718ac6666a436a1be1161c9"
uuid = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
version = "0.2.12"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PlutoVista]]
deps = ["ColorSchemes", "Colors", "GridVisualize", "UUIDs"]
git-tree-sha1 = "71c74291d06b88eccd593b7df16d4fb118d8910b"
uuid = "646e1f28-b900-46d7-9d87-d554eb38a413"
version = "0.3.2"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "1700b86ad59348c0f9f68ddc95117071f947072d"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[WriteVTK]]
deps = ["Base64", "CodecZlib", "FillArrays", "LightXML", "TranscodingStreams"]
git-tree-sha1 = "cc6b182732e3e00c3942f4e84fe88f0ec46e89a7"
uuid = "64499a7a-5c06-52f2-abe2-ccb03c286192"
version = "1.10.1"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─5fd62983-cad4-4a72-97e3-51c512451753
# ╠═fa059506-0710-11ec-1bb4-5f0937e3b169
# ╟─582e3416-5d15-4d01-936a-bd2a1e6b7310
# ╟─e882b9a6-5804-4381-a09c-bf9df81d798a
# ╟─bf1bb705-bdcc-4e10-9711-747fbbebbaef
# ╟─d44e5d2f-5483-4336-b000-1376b29e1b6f
# ╟─5bb01852-d3f5-4512-b8b4-fb5442eaeb97
# ╟─f74bd3d6-ee33-4f77-9f3b-a53be134041d
# ╟─b5fbbd91-0484-4bd8-a34d-4bab2d0588d6
# ╠═bf4a310d-b12f-4f55-a367-39b06b3ee228
# ╟─ac3cb451-b811-4a34-958c-b8266b6c4f17
# ╟─df2b6a48-e82b-4894-961f-8cc251eb0a01
# ╟─84cc43bb-33bf-472e-be76-1a570f772e82
# ╟─4e5e4d65-2db6-48d2-b8ac-5b5eb5d505e7
# ╟─6243a27b-706f-4d4b-acbd-5431868e2b6f
# ╟─b87e20a7-2d97-430c-b18f-da849ee8ebc9
# ╟─038bcbb0-6cc5-4a20-908d-e2a44dd7ef9c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
