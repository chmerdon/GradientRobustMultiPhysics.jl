

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


# edge integral means
# used e.g. for interpolation into P2, P2B finite elements
function ensure_edge_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, AT::Type{<:AbstractAssemblyType}, exact_function::UserData{AbstractDataFunction}; items = [], time = time) where {FEType <: AbstractFiniteElement}

    xItemVolumes = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemDofs = Dofmap4AssemblyType(FE, AT)
    nitems = num_sources(xItemNodes)
    if items == []
        items = 1 : nitems
    end

    # compute exact edge means
    ncomponents = get_ncomponents(FEType)
    edgemeans = zeros(Float64,ncomponents,nitems)
    integrate!(edgemeans, FE.xgrid, AT, exact_function; items = items, time = time)
    for item in items
        for c = 1 : ncomponents
            # subtract edge mean value of P1 part
            for dof = 1 : 2
                edgemeans[c,item] -= Target[xItemDofs[(c-1)*3 + dof,item]] * xItemVolumes[item] / 6
            end
            # set P2 edge bubble such that edge mean is preserved
            Target[xItemDofs[3*c,item]] = 3 // 2 * edgemeans[c,item] / xItemVolumes[item]
        end
    end
end


"""
````
function interpolate!(Target::FEVectorBlock,
     AT::Type{<:AbstractAssemblyType},
     source_data::UserData{AbstractDataFunction};
     items = [],
     time = 0,
     verbosity::Int = 0)
````

Interpolates the given source_data into the finite elements space assigned to the Target FEVectorBlock with the specified AbstractAssemblyType
(usualy ON_CELLS). The optional time argument is only used if the source_data depends on time.
"""
function interpolate!(Target::FEVectorBlock,
     AT::Type{<:AbstractAssemblyType},
     source_data::UserData{AbstractDataFunction};
     items = [],
     time = 0,
     verbosity::Int = 0)
    if verbosity > 0
        println("\nINTERPOLATING")
        println("=============")
        println("     target = $(Target.name)")
        println("         AT = $AT")
        println("         FE = $(Target.FES.name) (ndofs = $(Target.FES.ndofs))")
    end
    if Target.FES.broken == true
        FEType = eltype(Target.FES)
        FESc = FESpace{FEType}(Target.FES.xgrid; dofmaps_needed = [CellDofs])
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
     time = 0,
     verbosity::Int = 0)
````

Interpolates the given source_data into the finite elements space assigned to the Target FEVectorBlock. The optional time argument
is only used if the source_data depends on time.
"""
function interpolate!(Target::FEVectorBlock, source_data::UserData{AbstractDataFunction}; verbosity::Int = 0, time = 0)
    if verbosity > 0
        println("\nINTERPOLATING")
        println("=============")
        println("     target = $(Target.name)")
        println("         FE = $(Target.FES.name) (ndofs = $(Target.FES.ndofs))")
    end
    interpolate!(Target, ON_CELLS, source_data; time = time)
end



function nodevalues!(Target::AbstractArray{<:Real,2},
    Source::AbstractArray{<:Real,1},
    FE::FESpace,
    operator::Type{<:AbstractFunctionOperator} = Identity;
    regions::Array{Int,1} = [0],
    target_offset::Int = 0,
    source_offset::Int = 0,
    zero_target::Bool = true,
    continuous::Bool = false)

  xItemGeometries = FE.xgrid[CellGeometries]
  xItemRegions = FE.xgrid[CellRegions]
  xItemDofs = FE[CellDofs]
  xItemNodes = FE.xgrid[CellNodes]

  T = Base.eltype(Target)
  if regions == [0]
      try
          regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
      catch
          regions = [xItemRegions[1]]
      end        
  else
      regions = Array{Int32,1}(regions)    
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
    dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
    basisvals::Array{T,3} = basisevaler[1].cvals # pointer to operator results
    item::Int = 0
    nregions::Int = length(regions)
    ncomponents::Int = get_ncomponents(FEType)
    iEG::Int = 0
    node::Int = 0
    dof::Int = 0
    flag4node = zeros(Bool,nnodes)

    if zero_target
        fill!(Target, 0.0)
    end
    for item = 1 : nitems
        for r = 1 : nregions
        # check if item region is in regions
            if xItemRegions[item] == regions[r]

                # find index for CellType
                itemET = xItemGeometries[item]
                for j=1:length(EG)
                    if itemET == EG[j]
                        iEG = j
                        break;
                    end
                end

                # update FEbasisevaler
                update!(basisevaler[iEG],item)
                basisvals = basisevaler[iEG].cvals

                for i in eachindex(qf[iEG].w) # vertices
                    node = xItemNodes[i,item]
                    if continuous == false || flag4node[node] == false
                        nneighbours[node] += 1
                        for k = 1 :  size(basisvals,1)
                            for dof_i = 1 : ndofs4EG[iEG]
                                dof = xItemDofs[dof_i,item]
                                Target[k+target_offset,node] += Source[source_offset + dof] * basisvals[k,dof_i,i]
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