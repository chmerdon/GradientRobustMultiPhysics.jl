

#########################
# COMMON INTERPOLATIONS #
#########################

function slice(VTA::VariableTargetAdjacency, items = [], only_unique::Bool = true)
    subitems = zeros(Int,0)
    if items == []
        items = 1 : num_sources(VTA)
    end
    for item in items
        append!(subitems, VTA[:,item])
    end
    if only_unique
        subitems = unique(subitems)
    end
    return subitems
end

function slice(VTA::Array{<:Signed,2}, items = [], only_unique::Bool = true)
    subitems = zeros(Int,0)
    if items == []
        items = 1 : size(VTA,2)
    end
    for item in items
        append!(subitems, VTA[:,item])
    end
    if only_unique
        subitems = unique(subitems)
    end
    return subitems
end

# point evaluation (at vertices of geometry)
# for lowest order degrees of freedom
# used e.g. for interpolation into P1, P2, P2B, MINI finite elements
function point_evaluation!(Target::AbstractArray{T,1}, FES::FESpace{Tv, Ti, FEType, APT}, ::Type{AT_NODES}, exact_function::UserData{AbstractDataFunction}; items = [], component_offset::Int = 0, time = 0) where {T,Tv, Ti, FEType <: AbstractH1FiniteElement, APT}
    xCoordinates = FES.xgrid[Coordinates]
    xdim = size(xCoordinates,1)
    nnodes = size(xCoordinates,2)
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : nnodes
    end
    result = zeros(T,ncomponents)
    offset4component = 0:component_offset:ncomponents*component_offset
    # interpolate at nodes
    x = zeros(T,xdim)
    for j in items
        for k=1:xdim
            x[k] = xCoordinates[k,j]
        end    
        eval_data!(result, exact_function , x, time)
        for k = 1 : ncomponents
            Target[j+offset4component[k]] = result[k]
        end    
    end
end


# point evaluation (at vertices of geometry)
# for lowest order degrees of freedom
# used e.g. for interpolation into P1, P2, P2B, MINI finite elements
function point_evaluation!(Target::AbstractArray{T,1}, FES::FESpace{Tv, Ti, FEType, APT}, ::Type{AT_NODES}, exact_function::UserData{AbstractExtendedDataFunction}; items = [], component_offset::Int = 0, time = 0) where {T,Tv, Ti, FEType <: AbstractH1FiniteElement, APT}
    xCoordinates = FES.xgrid[Coordinates]
    xdim = size(xCoordinates,1)
    nnodes = size(xCoordinates,2)
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : nnodes
    end
    result = zeros(T,ncomponents)
    offset4component = 0:component_offset:ncomponents*component_offset
    # interpolate at nodes
    x = zeros(T,xdim)
    xNodeCells = atranspose(FES.xgrid[CellNodes])
    cell::Int = 0
    for j in items
        for k=1:xdim
            x[k] = xCoordinates[k,j]
        end    
        cell = xNodeCells[1,j]
        eval_data!(result, exact_function , x, time, nothing, cell, nothing)
        for k = 1 : ncomponents
            Target[j+offset4component[k]] = result[k]
        end    
    end
end

function point_evaluation_broken!(Target::AbstractArray{T,1}, FES::FESpace{Tv, Ti, FEType, APT}, ::Type{ON_CELLS}, exact_function::UserData{AbstractDataFunction}; items = [], time = 0) where {T,Tv, Ti, FEType <: AbstractH1FiniteElement, APT}
    xCoordinates = FES.xgrid[Coordinates]
    xdim = size(xCoordinates,1)
    xCellNodes = FES.xgrid[CellNodes]
    xCellDofs = FES[CellDofs]

    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : num_sources(xCellNodes)
    end
    result = zeros(T,ncomponents)
    nnodes_on_cell::Int = 0
    # interpolate at nodes
    x = zeros(T,xdim)
    for cell in items
        nnodes_on_cell = num_targets(xCellNodes, cell)
        for n = 1 : nnodes_on_cell
            j = xCellNodes[n,cell]
            for k=1:xdim
                x[k] = xCoordinates[k,j]
            end    
            eval_data!(result, exact_function , x, time)
            for k = 1 : ncomponents
                Target[xCellDofs[1,cell]+n-1+(k-1)*nnodes_on_cell] = result[k]
            end    
         end
    end
end


# # fall back function that is used if element does not define the cell moments directly (see below)
# # (so far only needed for H1-conforming elements with interpolations that preserve cell means)
# function get_ref_cellmoments(FEType::Type{<:AbstractFiniteElement}, EG::Type{<:AbstractElementGeometry}, AT::Type{<:AssemblyType} = ON_CELLS)
#     ndofs = get_ndofs(AT, FEType, EG)
#     ncomponents = get_ncomponents(FEType)
#     cellmoments = zeros(Float64, ndofs, ncomponents)
#     refbasis = get_basis(AT, FEType, EG)
#     ref_integrate!(cellmoments, EG, get_polynomialorder(FEType, EG), refbasis)
#     return cellmoments[:,1]
# end

# function ensure_cell_moments!(Target::AbstractArray{T,1}, FE::FESpace{Tv, Ti, FEType, APT}, exact_function!; nodedofs::Bool = true, facedofs::Int = 0, edgedofs::Int = 0, items = [], time = 0) where {T,Tv, Ti, FEType <: AbstractH1FiniteElement, APT}

#     # note: assumes that cell dof is always the last one and that there are no higher oder cell moments

#     xgrid = FE.xgrid
#     xItemVolumes = xgrid[CellVolumes]
#     xItemNodes = xgrid[CellNodes]
#     xItemDofs = FE[CellDofs]
#     xCellGeometries = xgrid[CellGeometries]
#     ncells = num_sources(xItemNodes)
#     nnodes = size(xgrid[Coordinates],2)
#     ncomponents = get_ncomponents(FEType)
#     if items == []
#         items = 1 : ncells
#     end

#     # compute referencebasis cell moments
#     uniqueEG = xgrid[UniqueCellGeometries]
#     cell_moments = Array{Array{T,1},1}(undef, length(uniqueEG))
#     for j = 1 : length(uniqueEG)
#         cell_moments[j] = get_ref_cellmoments(FEType,uniqueEG[j])
#     end

#     # compute exact cell integrals
#     cellintegrals = zeros(T,ncomponents,ncells)
#     integrate!(cellintegrals, xgrid, ON_CELLS, exact_function!; items = items, time = 0)
#     cellEG = uniqueEG[1]
#     nitemnodes::Int = num_nodes(cellEG)
#     nitemfaces::Int = num_faces(cellEG)
#     nitemedges::Int = num_edges(cellEG)
#     offset::Int = nitemnodes*nodedofs + nitemfaces*facedofs + nitemedges*edgedofs + 1
#     cellmoments::Array{T,1} = cell_moments[1]
#     iEG::Int = 1
#     for item in items
#         if length(uniqueEG) > 1
#             if cellEG != xCellGeometries[item]
#                 iEG = findfirst(isequal(cellEG), EG)
#                 cellEG = xCellGeometries[item]
#                 nitemnodes = num_nodes(cellEG)
#                 nitemfaces = num_faces(cellEG)
#                 nitemedges = num_edges(cellEG)
#                 offset = nitemnodes*nodedofs + nitemfaces*facedofs + nitemedges*edgedofs + 1
#                 cellmoments = cell_moments[iEG]
#             end
#         end
#         for c = 1 : ncomponents
#             # subtract integral of lower order dofs
#             for dof = 1 : offset - 1
#                 cellintegrals[c,item] -= Target[xItemDofs[(c-1)*offset + dof,item]] * xItemVolumes[item] * cellmoments[dof]
#             end
#             # set cell bubble such that cell mean is preserved
#             Target[xItemDofs[c*offset,item]] = cellintegrals[c,item] / (cellmoments[offset] * xItemVolumes[item])
#         end
#     end
# end

# # edge integral means
# # used e.g. for interpolation into P2, P2B finite elements
# function ensure_edge_moments!(Target::AbstractArray{T,1}, FE::FESpace{Tv, Ti, FEType, APT}, AT::Type{<:AssemblyType}, exact_function::UserData{AbstractDataFunction}; order = 0, items = [], time = time) where {T, Tv, Ti, FEType <: AbstractH1FiniteElement, APT}
#     xItemVolumes::Array{Tv,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
#     xItemNodes::Adjacency{Ti} = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
#     xItemDofs::DofMapTypes{Ti} = Dofmap4AssemblyType(FE, AT)

#     nitems::Int = num_sources(xItemNodes)
#     if items == []
#         items = 1 : nitems
#     end
#     ncomponents::Int = get_ncomponents(FEType)
#     edim::Int = get_edim(FEType)

#     # integrate moments of exact_function over edges
#     edgemoments::Array{Tv,2} = zeros(Tv,ncomponents,nitems)
#     if order == 0
#         nmoments = 1
#         mfactor = [1//6; 1//6] # = integral of nodal basis functions over edge
#         invdfactor = [3//2;] # = inverse integral of edge bubble over edge
#         coffset = 3
#     elseif order == 1
#         nmoments = 2
#         mfactor = 9//2 .* [-1//180 -2//135; 2//135 1//180] # = moments of nodal basis functions over edge
#         dfactor = 27//2 .* [-1//270 -7//540; 7//540 1//270]' # = integrals of interior edge functions over edge
#         invdfactor = inv(dfactor)
#         coffset = 4
#     end

#     function edgemoments_eval(m)
#         function closure(result, x, xref)
#             eval_data!(result, exact_function, x, time)
#             if order == 1
#                 result .*= (xref[1] - m//3)
#             end
#         end   
#     end   
#     for item in items
#         for n = 1 : nmoments, c = 1 : ncomponents
#             Target[xItemDofs[coffset*(c-1)+2+n,item]] = 0
#         end
#     end
#     for m = 1 : nmoments
#         edata_function = ExtendedDataFunction(edgemoments_eval(m), [ncomponents, edim]; dependencies = "XL", quadorder = exact_function.quadorder+1)
#         integrate!(edgemoments, FE.xgrid, AT, edata_function; items = items)
#         for item in items
#             for c = 1 : ncomponents
#                 # subtract edge mean value of P1 part
#                 for dof = 1 : 2
#                     edgemoments[c,item] -= Target[xItemDofs[(c-1)*coffset + dof,item]] * xItemVolumes[item] * mfactor[dof, m]
#                 end
#                 # set P2 edge bubble such that edge mean is preserved
#                 for n = 1 : nmoments
#                     Target[xItemDofs[coffset*(c-1)+2+n,item]] += invdfactor[n,m] * edgemoments[c,item] / xItemVolumes[item]
#                 end
#             end
#         end
#         fill!(edgemoments,0)
#     end
# end

# sets interior dofs sucht that the specified order of moments (e.g. integral test with Pk) are ensured
# while all dofs on the exterior are fixed, by solving a local problem on each cell (all with the same mass matrix)
#
# used for interpolation operators of elements with interior degrees of freedom (after setting the exterior ones with other methods)
# e.g. H1P2 ON_EDGES, H1MINI ON_CELLS, H1P2B ON_EDGES, ON_FACES, ON_CELLS
function ensure_moments!(Target::AbstractArray{T,1}, FE::FESpace{Tv, Ti, FEType, APT}, AT::Type{<:AssemblyType}, exact_function::UserData{AbstractDataFunction}; FEType_ref = "auto", order = 0, items = [], time = time) where {T, Tv, Ti, FEType <: AbstractH1FiniteElement, APT}

    xItemVolumes::Array{Tv,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemNodes::Adjacency{Ti} = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemDofs::DofMapTypes{Ti} = Dofmap4AssemblyType(FE, AT)
    EGs = FE.xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]

    bestapprox::Bool = false # if true interior dofs are set acoording to a constrained bestapproximation, otherwise to preserve the moments up to order, might become a kwarg later

    @assert length(EGs) == 1 "ensure_moments! currently only works with grids with a single element geometry"
    EG = EGs[1]

    nitems::Int = num_sources(xItemNodes)
    if items == []
        items = 1 : nitems
    end
    ncomponents::Int = get_ncomponents(FEType)
    edim::Int = dim_element(EG)
    order_FE = get_polynomialorder(FEType, EG)
    coffset::Int = get_ndofs(AT,FEType,EG) / ncomponents
    interior_offset = interior_dofs_offset(AT,FEType,EG)

    ## get basis for moments
    ## here we assume that the FEType looks like a H1Pk element on EG (which is true for all H1Pk elements)
    if order == 0
        ## the order of that element is order+1
        FEType_moments = H1P0{ncomponents}
    elseif order == 1
        FEType_moments = H1P1{ncomponents}
    elseif order == 2
        FEType_moments = H1P2{ncomponents,edim}
    else
        FEType_moments = H1Pk{ncomponents,edim,order}
    end

    if FEType_ref == "auto"
        if AT == ON_CELLS
            FEType_ref = FEType
        else
            if edim == 2 && order == 0
                FEType_ref = H1P2{ncomponents,edim}
            elseif edim == 1
                FEType_ref = H1Pk{ncomponents,edim,order+2}
            else
                @error "not yet supported"
            end
        end
    end

    moments_basis! = get_basis(ON_CELLS,FEType_moments,EG)
    nmoments::Int = get_ndofs_all(ON_CELLS,FEType_moments,EG)
    xgrid_ref = reference_domain(EG)
    nmoments4c::Int = nmoments / ncomponents
    idofs = zeros(Int,0)
    for c = 1 : ncomponents, m = 1 : nmoments4c
        push!(idofs, (c-1)*coffset + interior_offset + m)
    end

    MOMxBASIS::Array{Float64,2} = zeros(Float64,0,0)
    if (bestapprox) # interior dofs are set by best-approximation
        FE_onref = FESpace{FEType_ref}(xgrid_ref)
        MOMxBASIS_BLF = SymmetricBilinearForm(Float64,ON_CELLS,[FE_onref,FE_onref],[Identity,Identity])
        FEMMOMxBASIS = FEMatrix{Float64}("FExMOMENTS matrix",FE_onref,FE_onref)
        assemble!(FEMMOMxBASIS[1],MOMxBASIS_BLF)
        MOMxBASIS = FEMMOMxBASIS.entries' ./ xgrid_ref[CellVolumes][1]

        ## extract quadratic matrix for interior dofs
        MOMxINTERIOR = zeros(length(idofs),length(idofs))
        for j = 1 : length(idofs), k = 1 : length(idofs)
            MOMxINTERIOR[j,k] = MOMxBASIS[idofs[j],idofs[k]]
        end
        moments_eval = zeros(Float64,size(MOMxBASIS,1),ncomponents)
        moments_basis! = get_basis(ON_CELLS,FEType_ref,EG)
        MOMxBASIS = MOMxBASIS[:,idofs]
    else # interior dofs are set by moments
        ## calculate moments times basis functions
        FES_moments = FESpace{FEType_moments}(xgrid_ref)
        FE_onref = FESpace{FEType_ref}(xgrid_ref)
        MOMxBASIS_BLF = BilinearForm(Float64,ON_CELLS,[FES_moments,FE_onref],[Identity,Identity])
        FEMMOMxBASIS = FEMatrix{Float64}("FExMOMENTS matrix",FES_moments,FE_onref)
        assemble!(FEMMOMxBASIS[1],MOMxBASIS_BLF)
        MOMxBASIS = FEMMOMxBASIS.entries' ./ xgrid_ref[CellVolumes][1]

        ## extract quadratic matrix for interior dofs
        MOMxINTERIOR = zeros(length(idofs),size(MOMxBASIS,2))
        for j = 1 : length(idofs), k = 1 : size(MOMxBASIS,2)
            MOMxINTERIOR[j,k] = MOMxBASIS[idofs[j],k]
        end
        moments_eval = zeros(Float64,nmoments,ncomponents)
    end

    ### get permutation of dofs on reference EG and real cells
    subset_handler = get_basissubset(AT, FE, EG)
    current_subset = Array{Int,1}(1:size(MOMxBASIS,1))
    doforder_ref::Array{Int,1} = FE_onref[CellDofs][:,1]
    invA::Array{Float64,2} = inv(MOMxINTERIOR)

    ## evaluator for moments of exact function
    f_eval = zeros(Float64,ncomponents)
    function f_times_moments(result, x, xref)
        fill!(moments_eval,0)
        eval_data!(f_eval, exact_function, x, time)
        fill!(result,0)
        if (bestapprox)
            moments_basis!(moments_eval,xref)
            for m = 1 : nmoments, k = 1 : ncomponents
                result[m] += f_eval[k]*moments_eval[idofs[m],k]
            end
        else
            moments_basis!(moments_eval,xref)
            for m = 1 : nmoments, k = 1 : ncomponents
                result[m] += f_eval[k]*moments_eval[m,k]
            end
        end
        return nothing
    end   

    # integrate moments of exact_function over edges
    edgemoments::Array{T,2} = zeros(T,nmoments,nitems)
    xdim = size(FE.xgrid[Coordinates],1)
    edata_function = ExtendedDataFunction(f_times_moments, [nmoments, xdim]; dependencies = "XL", quadorder = exact_function.quadorder + (bestapprox ? order_FE : order))
    integrate!(edgemoments, FE.xgrid, AT, edata_function; items = items)

    localdof::Int = 0
    for item::Int in items
        if subset_handler != NothingFunction
            subset_handler(current_subset, item)
        end

        for m::Int = 1 : nmoments, exdof = 1 : interior_offset, c = 1 : ncomponents
            localdof = coffset*(c-1)+exdof
            edgemoments[m,item] -= Target[xItemDofs[localdof,item]] * MOMxBASIS[doforder_ref[current_subset[localdof]],m] * xItemVolumes[item]
        end
        for m::Int = 1 : nmoments
            localdof = idofs[m]
            Target[xItemDofs[localdof,item]] = 0
            for n::Int = 1 : nmoments
                Target[xItemDofs[localdof,item]] += invA[n,m] * edgemoments[n,item]
            end
            Target[xItemDofs[localdof,item]] /= xItemVolumes[item]
        end
    end
    return nothing
end

# remap boundary face interpolation to faces by using BFaceFaces (if there is no special function by the finite element defined)
function interpolate!(Target::FEVectorBlock, FES::FESpace, ::Type{ON_BFACES}, source_data; items = items, time = time)
    interpolate!(Target, FES, ON_FACES, source_data; items = FES.xgrid[BFaceFaces][items], time = time)
end

"""
````
function interpolate!(Target::FEVectorBlock,
     AT::Type{<:AssemblyType},
     source_data::UserData{AbstractDataFunction};
     items = [],
     time = 0)
````

Interpolates the given source_data into the finite elements space assigned to the Target FEVectorBlock with the specified AssemblyType
(usualy ON_CELLS). The optional time argument is only used if the source_data depends on time.
"""
function interpolate!(Target::FEVectorBlock{T,Tv,Ti},
     AT::Type{<:AssemblyType},
     source_data::UserData{<:AbstractDataFunction};
     items = [],
     time = 0) where {T,Tv,Ti}

    if is_timedependent(source_data)
        @logmsg MoreInfo "Interpolating $(source_data.name) >> $(Target.name) ($AT at time $time)"
    else
        @logmsg MoreInfo "Interpolating $(source_data.name) >> $(Target.name) ($AT)"
    end
    FEType = eltype(Target.FES)
    if Target.FES.broken == true
        FESc = FESpace{FEType}(Target.FES.xgrid)
        Targetc = FEVector{T}("auxiliary data",FESc)
        interpolate!(Targetc[1], FESc, AT, source_data; items = items, time = time)
        xCellDofs = Target.FES[CellDofs]
        xCellDofsc = FESc[CellDofs]
        dof::Int = 0
        dofc::Int = 0
        if items == []
            items = 1 : num_sources(xCellDofs)
        end
        for cell in items
            for k = 1 : num_targets(xCellDofs,cell)
                dof = xCellDofs[k,cell]
                dofc = xCellDofsc[k,cell]
                Target[dof] = Targetc.entries[dofc]
            end
        end
    else
        interpolate!(Target, Target.FES, AT, source_data; items = items, time = time)
    end
end

"""
````
function interpolate!(Target::FEVectorBlock,
     source_data::UserData{AbstractDataFunction};
     items = [],
     time = 0)
````

Interpolates the given source_data into the finite element space assigned to the Target FEVectorBlock. The optional time argument
is only used if the source_data depends on time.
"""
function interpolate!(Target::FEVectorBlock, source_data::UserData{<:AbstractDataFunction}; time = 0)
    interpolate!(Target, ON_CELLS, source_data; time = time)
end


"""
````
function interpolate!(Target::FEVectorBlock,
     source_data::FEVectorBlock;
     items = [])
````

Interpolates the given finite element function into the finite element space assigned to the Target FEVectorBlock. 
(Currently not the most efficient way as it is based on the PointEvaluation pattern and cell search.)
"""
function interpolate!(Target::FEVectorBlock{T1,Tv,Ti}, source_data::FEVectorBlock{T2,Tv,Ti}; operator = Identity, xtrafo = nothing, items = [], not_in_domain_value = 1e30, use_cellparents::Bool = false) where {T1,T2,Tv,Ti}
    # wrap point evaluation into function that is put into normal interpolate!
    xgrid = source_data.FES.xgrid
    xdim_source::Int = size(xgrid[Coordinates],1)
    xdim_target::Int = size(Target.FES.xgrid[Coordinates],1)
    if xdim_source != xdim_target
        @assert xtrafo !== nothing "grids have different coordinate dimensions, need xtrafo!"
    end
    FEType = eltype(source_data.FES)
    ncomponents::Int = get_ncomponents(FEType)
    resultdim::Int = Length4Operator(operator,xdim_source,ncomponents)
    PE = PointEvaluator(source_data, operator)
    xref = zeros(Tv,xdim_source)
    x_source = zeros(Tv,xdim_source)
    cell::Int = 1
    lastnonzerocell::Int = 1
    same_cells::Bool = xgrid == Target.FES.xgrid
    CF::CellFinder{Tv,Ti} = CellFinder(xgrid)

    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(ON_CELLS)]
    quadorder::Int = get_polynomialorder(FEType,EG[1])
    for j = 2 : length(EG)
        quadorder = max(quadorder, get_polynomialorder(FEType,EG[j]))
    end

    if same_cells || use_cellparents == true
        xCellParents::Array{Ti,1} = Target.FES.xgrid[CellParents]
        function point_evaluation_parentgrid!(result, x, target_cell)
            if same_cells
                lastnonzerocell = target_cell
            elseif use_cellparents
                lastnonzerocell = xCellParents[target_cell]
            end
            if xtrafo !== nothing
                xtrafo(x_source, x)
                cell = gFindLocal!(xref, CF, x_source; icellstart = lastnonzerocell)
            else
                cell = gFindLocal!(xref, CF, x; icellstart = lastnonzerocell)
            end
            evaluate!(result,PE,xref,cell)
            return nothing
        end
        fe_function = ExtendedDataFunction(point_evaluation_parentgrid!, [resultdim, xdim_target]; dependencies = "XI", quadorder = quadorder)
    else
        function point_evaluation_arbitrarygrids!(result, x)
            if xtrafo !== nothing
                xtrafo(x_source, x)
                cell = gFindLocal!(xref, CF, x_source; icellstart = lastnonzerocell)
                if cell == 0
                    cell = gFindBruteForce!(xref, CF, x_source)
                end
            else
                cell = gFindLocal!(xref, CF, x; icellstart = lastnonzerocell)
                if cell == 0
                    cell = gFindBruteForce!(xref, CF, x)
                end
            end
            if cell == 0
                fill!(result, not_in_domain_value)
            else
                evaluate!(result,PE,xref,cell)
                lastnonzerocell = cell
            end
            return nothing
        end
        fe_function = DataFunction(point_evaluation_arbitrarygrids!, [resultdim, xdim_target]; dependencies = "X", quadorder = quadorder)
    end
    interpolate!(Target, ON_CELLS, fe_function; items = items)
end



"""
````
function nodevalues!(
    Target::AbstractArray{<:Real,2},
    Source::AbstractArray{T,1},
    FE::FESpace{Tv,Ti,FEType,AT},
    operator::Type{<:AbstractFunctionOperator} = Identity;
    regions::Array{Int,1} = [0],
    abs::Bool = false,
    factor = 1,
    target_offset::Int = 0,   # start to write into Target after offset
    zero_target::Bool = true, # target vector is zeroed
    continuous::Bool = false)
````

Evaluates the finite element function with the coefficient vector Source (interpreted as a coefficient vector for the FESpace FE)
and the specified FunctionOperator at all the nodes of the (specified regions of the) grid and writes the values into Target.
Discontinuous (continuous = false) quantities are averaged.
"""
function nodevalues!(Target::AbstractArray{T,2},
    Source::AbstractArray{T,1},
    FE::FESpace{Tv,Ti,FEType,AT},
    operator::Type{<:AbstractFunctionOperator} = Identity;
    abs::Bool = false,
    factor = 1,
    regions::Array{Int,1} = [0],
    target_offset::Int = 0,
    source_offset::Int = 0,
    zero_target::Bool = true,
    continuous::Bool = false) where {T, Tv, Ti, FEType, AT}

    xItemGeometries = FE.xgrid[CellGeometries]
    xItemRegions::GridRegionTypes{Ti} = FE.xgrid[CellRegions]
    xItemDofs::DofMapTypes{Ti} = FE[CellDofs]
    xItemNodes::Adjacency{Ti} = FE.xgrid[CellNodes]

    if regions == [0]
        try
            regions = Array{Int,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end
    end

    # setup basisevaler for each unique cell geometries
    EG = FE.xgrid[UniqueCellGeometries]
    ndofs4EG::Array{Int,1} = Array{Int,1}(undef,length(EG))
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler::Array{FEBasisEvaluator{T,Tv,Ti},1} = Array{FEBasisEvaluator{T,Tv,Ti},1}(undef,length(EG))
    for j = 1 : length(EG)
        qf[j] = VertexRule(EG[j])
        basisevaler[j] = FEBasisEvaluator{T,EG[j],operator,ON_CELLS}(FE, qf[j])
        ndofs4EG[j] = size(basisevaler[j].cvals,2)
    end    
    cvals_resultdim::Int = size(basisevaler[1].cvals,1)
    target_resultdim::Int = abs ? 1 : cvals_resultdim
    @assert size(Target,1) >= target_resultdim "too small target dimension"

    nitems::Int = num_sources(xItemDofs)
    nnodes::Int = num_sources(FE.xgrid[Coordinates])
    nneighbours = zeros(Int,nnodes)
    basisvals::Array{T,3} = basisevaler[1].cvals # pointer to operator results
    item::Int = 0
    itemET = EG[1]
    nregions::Int = length(regions)
    iEG::Int = 1
    node::Int = 0
    dof::Ti = 0
    flag4node::Array{Bool,1} = zeros(Bool,nnodes)
    temp::Array{T,1} = zeros(T,cvals_resultdim)
    localT::Array{T,1} = zeros(T,cvals_resultdim)
    weights::Array{T,1} = qf[1].w

    if zero_target
        fill!(Target, 0)
    end
    for item = 1 : nitems
        for r = 1 : nregions
        # check if item region is in regions
            if xItemRegions[item] == regions[r]

                # find index for CellType
                if length(EG) > 1
                    itemET = xItemGeometries[item]
                    for j=1:length(EG)
                        if itemET == EG[j]
                            iEG = j
                            break;
                        end
                    end
                    weights = qf[iEG].w
                end

                # update FEbasisevaler
                update_febe!(basisevaler[iEG],item)
                basisvals = basisevaler[iEG].cvals

                for i in eachindex(weights) # vertices
                    node = xItemNodes[i,item]
                    fill!(localT,0)
                    if continuous == false || flag4node[node] == false
                        nneighbours[node] += 1
                        flag4node[node] = true
                        begin
                            for dof_i = 1 : ndofs4EG[iEG]
                                dof = xItemDofs[dof_i,item]
                                eval_febe!(temp, basisevaler[iEG], dof_i, i)
                                for k = 1 : cvals_resultdim
                                    localT[k] += Source[source_offset + dof] * temp[k]
                                    #Target[k+target_offset,node] += temp[k] * Source[source_offset + dof]
                                end
                            end
                        end
                        localT .*= factor
                        if abs
                            for k = 1 : cvals_resultdim
                                Target[1+target_offset,node] += localT[k]^2
                            end
                        else
                            for k = 1 : cvals_resultdim
                                Target[k+target_offset,node] += localT[k]
                            end
                        end
                    end
                end  
                break; # region for loop
            end # if in region    
        end # region for loop
    end # item for loop

    if continuous == false
        for node = 1 : nnodes, k = 1 : target_resultdim
            Target[k+target_offset,node] /= nneighbours[node]
        end
    end

    if abs
        for node = 1 : nnodes
            Target[1+target_offset,node] = sqrt(Target[1+target_offset,node])
        end
    end

    return nothing
end


"""
````
function nodevalues!(
    Target::AbstractArray{<:Real,2},
    Source::FEVectorBlock,
    operator::Type{<:AbstractFunctionOperator} = Identity;
    regions::Array{Int,1} = [0],
    abs::Bool = false,
    factor = 1,
    target_offset::Int = 0,   # start to write into Target after offset
    zero_target::Bool = true, # target vector is zeroed
    continuous::Bool = false)
````

Evaluates the finite element function with the coefficient vector Source
and the specified FunctionOperator at all the nodes of the (specified regions of the) grid and writes the values into Target.
Discontinuous (continuous = false) quantities are averaged.
"""
function nodevalues!(Target, Source::FEVectorBlock, operator::Type{<:AbstractFunctionOperator} = Identity; regions::Array{Int,1} = [0], abs::Bool = false, factor = 1, continuous::Bool = false, target_offset::Int = 0, zero_target::Bool = true)
    nodevalues!(Target, Source.entries, Source.FES, operator; regions = regions, continuous = continuous, source_offset = Source.offset, abs = abs, factor = factor, zero_target = zero_target, target_offset = target_offset)
end


"""
````
function nodevalues(
    Source::FEVectorBlock,
    operator::Type{<:AbstractFunctionOperator} = Identity;
    regions::Array{Int,1} = [0],
    abs::Bool = false,
    factor = 1,
    target_offset::Int = 0,   # start to write into Target after offset
    zero_target::Bool = true, # target vector is zeroed
    continuous::Bool = false)
````

Evaluates the finite element function with the coefficient vector Source
and the specified FunctionOperator at all the nodes of the (specified regions of the)
grid and returns an array with the values.
Discontinuous (continuous = false) quantities are averaged.
"""
function nodevalues(Source::FEVectorBlock{T,Tv,Ti,FEType,APT}, operator::Type{<:AbstractFunctionOperator} = Identity; abs::Bool = false, regions::Array{Int,1} = [0], factor = 1, continuous = "auto") where {T,Tv,Ti,APT,FEType}
    if continuous == "auto"
        if FEType <: AbstractH1FiniteElement && operator == Identity
            continuous = true
        else
            continuous = false
        end
    end
    if abs
        nvals = 1
    else
        xdim = size(Source.FES.xgrid[Coordinates],2)
        ncomponents = get_ncomponents(eltype(Source.FES))
        nvals = Length4Operator(operator, xdim, ncomponents)
    end
    Target = zeros(T,nvals,num_nodes(Source.FES.xgrid))
    nodevalues!(Target, Source.entries, Source.FES, operator; regions = regions, continuous = continuous, source_offset = Source.offset, factor = factor, abs = abs)
    return Target
end

"""
````
function nodevalues_view(
    Source::FEVectorBlock,
    operator::Type{<:AbstractFunctionOperator} = Identity)
````

Returns a vector of views of the nodal values of the Source block (currently works for unbroken H1-conforming elements) that directly accesses the coefficients.
"""
function nodevalues_view(Source::FEVectorBlock{T,Tv,Ti,FEType,APT}, operator::Type{<:AbstractFunctionOperator} = Identity) where {T,Tv,Ti,APT,FEType}

    if (FEType <: AbstractH1FiniteElement) && (operator == Identity) && (Source.FES.broken == false)
        # give a direct view without computing anything
        ncomponents = get_ncomponents(FEType)
        array_of_views = []
        offset::Int = Source.offset
        coffset::Int = Source.FES.ndofs / ncomponents
        nnodes::Int = num_nodes(Source.FES.xgrid)
        for k = 1 : ncomponents
            push!(array_of_views,view(Source.entries,offset+1:offset+nnodes))
            offset += coffset
        end
        return array_of_views
    else
        @error "nodevalues_view node evalable for FEType = $FEType and operator = $operator"
    end
end





"""
````
function displace_mesh!(xgrid::ExtendableGrid, Source::FEVectorBlock; magnify = 1)
````
Moves all nodes of the grid by adding the displacement field in Source (expects a vector-valued finite element)
times a magnify value.
"""
function displace_mesh!(xgrid::ExtendableGrid, Source::FEVectorBlock; magnify = 1)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(eltype(xgrid[Coordinates]),get_ncomponents(Base.eltype(Source.FES)),nnodes)
    nodevalues!(nodevals, Source, Identity)
    xgrid[Coordinates] .+= magnify * nodevals

    # remove all keys from grid components that might have changed and need a reinstantiation
    delete!(xgrid.components,CellVolumes)
    delete!(xgrid.components,FaceVolumes)
    delete!(xgrid.components,EdgeVolumes)
    delete!(xgrid.components,FaceNormals)
    delete!(xgrid.components,EdgeTangents)
    delete!(xgrid.components,BFaceVolumes)
    delete!(xgrid.components,BEdgeVolumes)
end


function displace_mesh(xgrid::ExtendableGrid, Source::FEVectorBlock; magnify = 1)
    xgrid_displaced = deepcopy(xgrid)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(eltype(xgrid[Coordinates]),get_ncomponents(Base.eltype(Source.FES)),nnodes)
    nodevalues!(nodevals, Source, Identity)
    xgrid_displaced[Coordinates] .+= magnify * nodevals
    return xgrid_displaced
end

