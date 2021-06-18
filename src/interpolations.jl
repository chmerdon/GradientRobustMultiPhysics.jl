

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
function point_evaluation!(Target::AbstractArray{<:Real,1}, FES::FESpace{FEType}, ::Type{AT_NODES}, exact_function::UserData{AbstractDataFunction}; items = [], component_offset::Int = 0, time = 0) where {FEType <: AbstractFiniteElement}
    xCoordinates = FES.xgrid[Coordinates]
    xdim = size(xCoordinates,1)
    nnodes = size(xCoordinates,2)
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : nnodes
    end
    result = zeros(Float64,ncomponents)
    offset4component = 0:component_offset:ncomponents*component_offset
    # interpolate at nodes
    x = zeros(Float64,xdim)
    for j in items
        for k=1:xdim
            x[k] = xCoordinates[k,j]
        end    
        eval!(result, exact_function , x, time)
        for k = 1 : ncomponents
            Target[j+offset4component[k]] = result[k]
        end    
    end
end

function point_evaluation_broken!(Target::AbstractArray{T,1}, FES::FESpace{FEType}, ::Type{ON_CELLS}, exact_function::UserData{AbstractDataFunction}; items = [], time = 0) where {FEType <: AbstractFiniteElement, T<:Real} 
    xCoordinates = FES.xgrid[Coordinates]
    xdim = size(xCoordinates,1)
    xCellNodes = FES.xgrid[CellNodes]
    xCellDofs = FES[CellDofs]

    nnodes = size(xCoordinates,2)
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
            eval!(result, exact_function , x, time)
            for k = 1 : ncomponents
                Target[xCellDofs[1,cell]+n-1+(k-1)*nnodes_on_cell] = result[k]
            end    
         end
    end
end


# fall back function that is used if element does not define the cell moments directly (see below)
# (so far only needed for H1-conforming elements with interpolations that preserve cell means)
function get_ref_cellmoments(FEType::Type{<:AbstractFiniteElement}, EG::Type{<:AbstractElementGeometry}, AT::Type{<:AbstractAssemblyType} = ON_CELLS)
    ndofs = get_ndofs(AT, FEType, EG)
    ncomponents = get_ncomponents(FEType)
    cellmoments = zeros(Float64, ndofs, ncomponents)
    refbasis = get_basis(AT, FEType, EG)
    ref_integrate!(cellmoments, EG, get_polynomialorder(FEType, EG), refbasis)
    return cellmoments[:,1]
end

function ensure_cell_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, exact_function!; nodedofs::Bool = true, facedofs::Int = 0, edgedofs::Int = 0, items = [], time = 0) where {FEType <: AbstractH1FiniteElement}
    # note: assumes that cell dof is always the last one
    xgrid = FE.xgrid
    xItemVolumes = xgrid[CellVolumes]
    xItemNodes = xgrid[CellNodes]
    xItemDofs = FE[CellDofs]
    xCellGeometries = xgrid[CellGeometries]
    ncells = num_sources(xItemNodes)
    nnodes = size(xgrid[Coordinates],2)
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : ncells
    end

    # compute referencebasis cell moments
    uniqueEG = xgrid[UniqueCellGeometries]
    cell_moments = Array{Array{Float64,1},1}(undef, length(uniqueEG))
    for j = 1 : length(uniqueEG)
        cell_moments[j] = get_ref_cellmoments(FEType,uniqueEG[j])
    end

    # compute exact cell integrals
    cellintegrals = zeros(Float64,ncomponents,ncells)
    integrate!(cellintegrals, xgrid, ON_CELLS, exact_function!; items = items, time = 0)
    cellEG = uniqueEG[1]
    nitemnodes::Int = nnodes_for_geometry(cellEG)
    nitemfaces::Int = nfaces_for_geometry(cellEG)
    nitemedges::Int = nedges_for_geometry(cellEG)
    offset::Int = nitemnodes*nodedofs + nitemfaces*facedofs + nitemedges*edgedofs + 1
    cellmoments::Array{Float64,1} = cell_moments[1]
    iEG::Int = 1
    for item in items
        if length(uniqueEG) > 1
            if cellEG != xCellGeometries[item]
                iEG = findfirst(isequal(cellEG), EG)
                cellEG = xCellGeometries[item]
                nitemnodes = nnodes_for_geometry(cellEG)
                nitemfaces = nfaces_for_geometry(cellEG)
                nitemedges = nedges_for_geometry(cellEG)
                offset = nitemnodes*nodedofs + nitemfaces*facedofs + nitemedges*edgedofs + 1
                cellmoments = cell_moments[iEG]
            end
        end
        for c = 1 : ncomponents
            # subtract integral of lower order dofs
            for dof = 1 : offset - 1
                cellintegrals[c,item] -= Target[xItemDofs[(c-1)*offset + dof,item]] * xItemVolumes[item] * cellmoments[dof]
            end
            # set cell bubble such that cell mean is preserved
            Target[xItemDofs[c*offset,item]] = cellintegrals[c,item] / (cellmoments[offset] * xItemVolumes[item])
        end
    end
end

# edge integral means
# used e.g. for interpolation into P2, P2B finite elements
function ensure_edge_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, AT::Type{<:AbstractAssemblyType}, exact_function::UserData{AbstractDataFunction}; order = 0, items = [], time = time) where {FEType <: AbstractH1FiniteElement}

    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemDofs = Dofmap4AssemblyType(FE, AT)

    nitems = num_sources(xItemNodes)
    if items == []
        items = 1 : nitems
    end
    ncomponents = get_ncomponents(FEType)
    edim = get_edim(FEType)

    # integrate moments of exact_function over edges
    edgemoments = zeros(Float64,ncomponents,nitems)
    if order == 0
        nmoments = 1
        mfactor = [1//6; 1//6] # = integral of nodal basis functions over edge
        invdfactor = [3//2;] # = inverse integral of edge bubble over edge
        coffset = 3
    elseif order == 1
        nmoments = 2
        mfactor = 9//2 .* [-1//180 -2//135; 2//135 1//180] # = moments of nodal basis functions over edge
        dfactor = 27//2 .* [-1//270 -7//540; 7//540 1//270]' # = integrals of interior edge functions over edge
        invdfactor = inv(dfactor)
        coffset = 4
    end
    function edgemoments_eval(m)
        function closure(result, x, xref)
            eval!(result, exact_function, x, time)
            if order == 1
                result .*= (xref[1] - m//3)
            end
        end   
    end   
    for item in items
        for n = 1 : nmoments, c = 1 : ncomponents
            Target[xItemDofs[coffset*(c-1)+2+n,item]] = 0
        end
    end
    for m = 1 : nmoments
        edata_function = ExtendedDataFunction(edgemoments_eval(m), [ncomponents, edim]; dependencies = "XL", quadorder = exact_function.quadorder+1)
        integrate!(edgemoments, FE.xgrid, AT, edata_function; items = items)
        for item in items
            for c = 1 : ncomponents
                # subtract edge mean value of P1 part
                for dof = 1 : 2
                    edgemoments[c,item] -= Target[xItemDofs[(c-1)*coffset + dof,item]] * xItemVolumes[item] * mfactor[dof, m]
                end
                # set P2 edge bubble such that edge mean is preserved
                for n = 1 : nmoments
                    Target[xItemDofs[coffset*(c-1)+2+n,item]] += invdfactor[n,m] * edgemoments[c,item] / xItemVolumes[item]
                end
            end
        end
        fill!(edgemoments,0)
    end
end

# remap boundary face interpolation to faces by using BFaces (if there is no special function by the finite element defined)
function interpolate!(Target::FEVectorBlock, FES::FESpace, ::Type{ON_BFACES}, source_data; items = items, time = time)
    interpolate!(Target, FES, ON_FACES, source_data; items = FES.xgrid[BFaces][items], time = time)
end

"""
````
function interpolate!(Target::FEVectorBlock,
     AT::Type{<:AbstractAssemblyType},
     source_data::UserData{AbstractDataFunction};
     items = [],
     time = 0)
````

Interpolates the given source_data into the finite elements space assigned to the Target FEVectorBlock with the specified AbstractAssemblyType
(usualy ON_CELLS). The optional time argument is only used if the source_data depends on time.
"""
function interpolate!(Target::FEVectorBlock,
     AT::Type{<:AbstractAssemblyType},
     source_data::UserData{AbstractDataFunction};
     items = [],
     time = 0)

    if is_timedependent(source_data)
        @logmsg MoreInfo "Interpolating $(source_data.name) >> $(Target.name) ($AT at time $time)"
    else
        @logmsg MoreInfo "Interpolating $(source_data.name) >> $(Target.name) ($AT)"
    end
    FEType = eltype(Target.FES)
    if Target.FES.broken == true
        FESc = FESpace{FEType}(Target.FES.xgrid)
        Targetc = FEVector{Float64}("auxiliary data",FESc)
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
function interpolate!(Target::FEVectorBlock, source_data::UserData{AbstractDataFunction}; time = 0)
    interpolate!(Target, ON_CELLS, source_data; time = time)
end


"""
````
function interpolate!(Target::FEVectorBlock,
     source_data::UserData{AbstractDataFunction};
     items = [])
````

Interpolates the given finite element function into the finite element space assigned to the Target FEVectorBlock. 
(Currently not the most efficient way as it is based on the PointEvaluation pattern and cell search.)
"""
function interpolate!(Target::FEVectorBlock, source_data::FEVectorBlock; operator = Identity, xtrafo = nothing, items = [], not_in_domain_value = 1e30)
    # wrap point evaluation into function that is put into normal interpolate!
    xgrid = source_data.FES.xgrid
    xdim_source::Int = size(xgrid[Coordinates],1)
    xdim_target::Int = size(Target.FES.xgrid[Coordinates],1)
    if xdim_source != xdim_target
        @assert xtrafo !== nothing "grids have different coordinate dimensions, need xtrafo!"
    end
    FEType = typeof(source_data.FES).parameters[1]
    ncomponents = get_ncomponents(FEType)
    resultdim = Length4Operator(operator,xdim_source,ncomponents)
    PE = PointEvaluator{Float64,FEType,xgrid[CellGeometries][1],operator,ON_CELLS}(source_data.FES, source_data)
    xref = zeros(Float64,xdim_source)
    x_source = zeros(Float64,xdim_source)
    cell::Int = 1
    lastnonzerocell::Int = 1
    function point_evaluation!(result, x)
        if xtrafo !== nothing
            xtrafo(x_source, x)
            cell = gFindLocal!(xref, xgrid, x_source; icellstart = lastnonzerocell)
            if cell == 0
                cell = gFindBruteForce!(xref, xgrid, x_source)
            end
        else
            cell = gFindLocal!(xref, xgrid, x; icellstart = lastnonzerocell)
            if cell == 0
                cell = gFindBruteForce!(xref, xgrid, x)
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
    fe_function = DataFunction(point_evaluation!, [resultdim, xdim_target]; dependencies = "X", quadorder = 2)
    interpolate!(Target, ON_CELLS, fe_function; items = items)
end



function nodevalues!(Target::AbstractArray{T,2},
    Source::AbstractArray{T,1},
    FE::FESpace,
    operator::Type{<:AbstractFunctionOperator} = Identity;
    regions::Array{Int,1} = [0],
    target_offset::Int = 0,
    source_offset::Int = 0,
    zero_target::Bool = true,
    continuous::Bool = false) where {T <: Real}

    xItemGeometries = FE.xgrid[CellGeometries]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE.xgrid[CellRegions]
    xItemDofs::DofMapTypes = FE[CellDofs]
    xItemNodes::GridAdjacencyTypes = FE.xgrid[CellNodes]

    if regions == [0]
        try
            regions = Array{Int,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int,1}(regions)    
    end

    # setup basisevaler for each unique cell geometries
    EG = FE.xgrid[UniqueCellGeometries]
    ndofs4EG = Array{Int,1}(undef,length(EG))
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{FEBasisEvaluator,1}(undef,length(EG))
    FEType = Base.eltype(FE)
    for j = 1 : length(EG)
        qf[j] = VertexRule(EG[j])
        basisevaler[j] = FEBasisEvaluator{T,FEType,EG[j],operator,ON_CELLS}(FE, qf[j])
        ndofs4EG[j] = size(basisevaler[j].cvals,2)
    end    
    cvals_resultdim::Int = size(basisevaler[1].cvals,1)
    @assert size(Target,1) >= cvals_resultdim

    nitems::Int = num_sources(xItemDofs)
    nnodes::Int = num_sources(FE.xgrid[Coordinates])
    nneighbours = zeros(Int,nnodes)
    basisvals::Array{T,3} = basisevaler[1].cvals # pointer to operator results
    item::Int = 0
    itemET = EG[1]
    nregions::Int = length(regions)
    ncomponents::Int = get_ncomponents(FEType)
    iEG::Int = 1
    node::Int = 0
    dof::Int = 0
    flag4node = zeros(Bool,nnodes)
    temp::Array{T,1} = zeros(T,cvals_resultdim)
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
                update!(basisevaler[iEG],item)
                basisvals = basisevaler[iEG].cvals

                for i in eachindex(weights) # vertices
                    node = xItemNodes[i,item]
                     if continuous == false || flag4node[node] == false
                        nneighbours[node] += 1
                        flag4node[node] = true
                        begin
                            for dof_i = 1 : ndofs4EG[iEG]
                                dof = xItemDofs[dof_i,item]
                                eval!(temp, basisevaler[iEG], dof_i, i)
                                for k = 1 : cvals_resultdim
                                    Target[k+target_offset,node] += temp[k] * Source[source_offset + dof]
                                end
                            end
                        end
                    end
                end  
                break; # region for loop
            end # if in region    
        end # region for loop
    end # item for loop

    if continuous == false
        for node = 1 : nnodes, k = 1 : cvals_resultdim
            Target[k+target_offset,node] /= nneighbours[node]
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
    target_offset::Int = 0,   # start to write into Target after offset
    zero_target::Bool = true, # target vector is zeroed
    continuous::Bool = false)
````

Evaluates the finite element function with the coefficient vector Source
and the specified FunctionOperator at all the nodes of the (specified regions of the) grid and writes the values into Target.
Discontinuous (continuous = false) quantities are averaged.
"""
function nodevalues!(Target::AbstractArray{<:Real,2}, Source::FEVectorBlock, operator::Type{<:AbstractFunctionOperator} = Identity; regions::Array{Int,1} = [0], continuous::Bool = false, target_offset::Int = 0, zero_target::Bool = true)
    nodevalues!(Target, Source.entries, Source.FES, operator; regions = regions, continuous = continuous, source_offset = Source.offset, zero_target = zero_target, target_offset = target_offset)
end


function displace_mesh!(xgrid::ExtendableGrid, Source::FEVectorBlock; magnify = 1)
    nnodes = size(xgrid[Coordinates],2)
    nodevals = zeros(eltype(xgrid[Coordinates]),get_ncomponents(Base.eltype(Source.FES)),nnodes)
    nodevalues!(nodevals, Source, Identity)
    xgrid[Coordinates] .+= magnify * nodevals
end

