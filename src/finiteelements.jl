#######################################################################################################
#######################################################################################################
### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
#######################################################################################################
#######################################################################################################

abstract type AbstractFiniteElement end
  

#############################
# Finite Element Supertypes #
#############################
#
# they are used to steer the kind of local2global transformation
# below subtypes are defined that define basis functions on reference geometries 
# and some other information like polyonomial degrees etc.

abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
abstract type AbstractH1FiniteElementWithCoefficients <: AbstractH1FiniteElement end
abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end

# dofmaps are stored as abstract grid adjacencies
abstract type DofMap <: AbstractGridAdjacency end
abstract type CellDofs <: DofMap end
abstract type FaceDofs <: DofMap end
abstract type EdgeDofs <: DofMap end
abstract type BFaceDofs <: DofMap end
abstract type BEdgeDofs <: DofMap end

UCG4DofMap(::Type{CellDofs}) = UniqueCellGeometries
UCG4DofMap(::Type{FaceDofs}) = UniqueFaceGeometries
UCG4DofMap(::Type{EdgeDofs}) = UniqueEdgeGeometries
UCG4DofMap(::Type{BFaceDofs}) = UniqueBFaceGeometries
UCG4DofMap(::Type{BEdgeDofs}) = UniqueBEdgeGeometries

SuperItemNodes4DofMap(::Type{CellDofs}) = CellNodes
SuperItemNodes4DofMap(::Type{FaceDofs}) = FaceNodes
SuperItemNodes4DofMap(::Type{EdgeDofs}) = EdgeNodes
SuperItemNodes4DofMap(::Type{BFaceDofs}) = FaceNodes
SuperItemNodes4DofMap(::Type{BEdgeDofs}) = EdgeNodes

ItemGeometries4DofMap(::Type{CellDofs}) = CellGeometries
ItemGeometries4DofMap(::Type{FaceDofs}) = FaceGeometries
ItemGeometries4DofMap(::Type{EdgeDofs}) = EdgeGeometries
ItemGeometries4DofMap(::Type{BFaceDofs}) = BFaceGeometries
ItemGeometries4DofMap(::Type{BEdgeDofs}) = BEdgeGeometries

ItemEdges4DofMap(::Type{CellDofs}) = CellEdges
ItemEdges4DofMap(::Type{FaceDofs}) = FaceEdges
ItemEdges4DofMap(::Type{BFaceDofs}) = FaceEdges

Sub2Sup4DofMap(::Type{BFaceDofs}) = BFaces
Sub2Sup4DofMap(::Type{BEdgeDofs}) = BEdges


"""
$(TYPEDEF)

A struct that has a finite element type as parameter and carries dofmaps (CellDofs, FaceDofs, BFaceDofs) plus additional grid information and access to arrays holding coefficients if needed.
"""
mutable struct FESpace{FEType<:AbstractFiniteElement}
    name::String                          # full name of finite element space (used in messages)
    ndofs::Int                            # total number of dofs
    xgrid::ExtendableGrid                 # link to xgrid 
    dofmaps::Dict{Type{<:AbstractGridComponent},Any} # backpack with dofmaps
end

function FESpace{FEType}(xgrid::ExtendableGrid; name = "", dofmaps_needed = "auto", verbosity = 0 ) where {FEType <:AbstractFiniteElement}
    # first generate some empty FESpace
    dummyVTA = VariableTargetAdjacency(Int32)
    FES = FESpace{FEType}(name,0,xgrid,Dict{Type{<:AbstractGridComponent},Any}())

    if verbosity > 0
        println("  Initialising FESpace $FEType...")
    end
    # then update data according to init specifications in FEdefinition files
    init!(FES)

    if dofmaps_needed == "auto"
        dofmaps_needed = [CellDofs, FaceDofs, BFaceDofs]
        if FEType <: AbstractHcurlFiniteElement && get_ncomponents(FEType) == 3
            push!(dofmaps_needed, EdgeDofs)
            push!(dofmaps_needed, BEdgeDofs)
        end
    end

    # generate required dof maps
    for j = 1 : length(dofmaps_needed)
        if verbosity > 0
            println("  ...generating dofmap for $(dofmaps_needed[j])")
            @time init_dofmap!(FES, dofmaps_needed[j])
        else
            init_dofmap!(FES, dofmaps_needed[j])
        end
    end

    return FES
end


"""
$(TYPEDSIGNATURES)

Custom `eltype` function for `FESpace` returns the finite element type of the finite element space.
"""
Base.eltype(::FESpace{FEType}) where {FEType<:AbstractFiniteElement} = FEType


"""
$(TYPEDSIGNATURES)

Custom `show` function for `FESpace` that prints some information and all available dofmaps.
"""
function Base.show(io::IO, FES::FESpace{FEType}) where {FEType<:AbstractFiniteElement}
	  println("\nFESpace information")
    println("===================")
    println("     name = $(FES.name)")
    println("   FEType = $FEType")
    println("  FEClass = $(supertype(FEType))")
    println("    ndofs = $(FES.ndofs)\n")
    println("")
    println("DofMaps");
    println("==========");
    for tuple in FES.dofmaps
        println("> $(tuple[1])")
    end
end

# default coefficient functions that can be overwritten by finite element that has non-default coeffcients
# ( see e.g. h1v_br.jl )
function get_coefficients_on_face!(FE::FESpace{<:AbstractFiniteElement}, ::Type{<:AbstractElementGeometry})
    function closure(coefficients, face)
        fill!(coefficients,1.0)
    end
end    
function get_coefficients_on_cell!(FE::FESpace{<:AbstractFiniteElement}, ::Type{<:AbstractElementGeometry})
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
    end
end    
function get_basissubset_on_cell!(FE::FESpace{<:AbstractFiniteElement}, ::Type{<:AbstractElementGeometry})
    function closure(subset_ids, cell)
        # it is assumed that subset_ids is already 1 vector of form subset_ids = 1:ndofs
        # meaning that all basis functions on the reference cells are used
        # see 3D implementation of BDM1 for an example how this is can be used the choose
        # different basis functions depending on the face orientations (which in 3D is not just a sign)
        return nothing
    end
end  
get_ndofs_on_cell_all(FEType::Type{<:AbstractFiniteElement}, EG::Type{<:AbstractElementGeometry}) = get_ndofs_on_cell(FEType, EG)






function init_dofmap_from_pattern!(FES::FESpace{FEType}, DM::Type{<:DofMap}) where {FEType <: AbstractFiniteElement}

    ## prepare dofmap patterns
    xgrid = FES.xgrid
    EG = xgrid[UCG4DofMap(DM)]
    ncomponents::Int = get_ncomponents(FEType)
    need_faces = false
    need_edges = false
    maxdofs4item::Int = 0
    dofmap_patterns = Array{String,1}(undef,length(EG))
    dofmap_quantifiers = Array{Array{Int,1},1}(undef,length(EG))
    dofs4item4component::Int = 0
    dofs4item_single::Int = 0
    for j = 1 : length(EG)
        pattern = get_dofmap_pattern(FEType, DM, EG[j])
        dofmap_patterns[j] = ""
        dofmap_quantifiers[j] = zeros(Int,Int(length(pattern)/2))
        for k = 1 : Int(length(pattern)/2)
            dofmap_patterns[j] *= pattern[2*k-1]
            dofmap_quantifiers[j][k] = parse(Int,pattern[2*k])
            if dofmap_patterns[j][k] == 'N'
                dofs4item4component += nnodes_for_geometry(EG[j])*dofmap_quantifiers[j][k]
            elseif dofmap_patterns[j][k] == 'F'
                dofs4item4component += nfaces_for_geometry(EG[j])*dofmap_quantifiers[j][k]
                need_faces = true
            elseif dofmap_patterns[j][k] == 'E'
                dofs4item4component += nedges_for_geometry(EG[j])*dofmap_quantifiers[j][k]
                need_edges = true
            elseif dofmap_patterns[j][k] == 'I'
                dofs4item4component += dofmap_quantifiers[j][k]
            elseif dofmap_patterns[j][k] == 'f'
                dofs4item_single += nfaces_for_geometry(EG[j])*dofmap_quantifiers[j][k]
                need_faces = true
            elseif dofmap_patterns[j][k] == 'i'
                dofs4item_single += dofmap_quantifiers[j][k]
                need_faces = true
            end
        end
        maxdofs4item = max(maxdofs4item,dofs4item4component*ncomponents + dofs4item_single)
        dofs4item4component = 0
        dofs4item_single = 0
    end

    nnodes = size(xgrid[Coordinates],2)
    ncells = num_sources(xgrid[CellNodes])
    xItemNodes = xgrid[SuperItemNodes4DofMap(DM)]
    xItemGeometries = xgrid[ItemGeometries4DofMap(DM)]
    if need_faces
        xItemFaces = xgrid[CellFaces]
        nfaces = num_sources(xgrid[FaceNodes])
    end
    if need_edges
        xItemEdges = xgrid[ItemEdges4DofMap(DM)]
        nedges = num_sources(xgrid[EdgeNodes])
    end


    offset4component::Int = 0
    xItemNodes = xgrid[SuperItemNodes4DofMap(DM)]
    nitems = num_sources(xItemNodes)
    for k = 1 : length(dofmap_patterns[1])
        if dofmap_patterns[1][k] == 'N'
            offset4component += nnodes*dofmap_quantifiers[1][k]
        elseif dofmap_patterns[1][k] == 'F'
            offset4component += nfaces*dofmap_quantifiers[1][k]
        elseif dofmap_patterns[1][k] == 'C'
            offset4component += ncells*dofmap_quantifiers[1][k]
        elseif dofmap_patterns[1][k] == 'E'
            offset4component += nedges*dofmap_quantifiers[1][k]
        elseif dofmap_patterns[1][k] == 'I'
            offset4component += nitems*dofmap_quantifiers[1][k]
        end
    end

    ## generate dofmap from patterns
    sub2sup = nothing
    nsubitems = 0
    try 
        Sub2SupItems = xgrid[Sub2Sup4DofMap(DM)]
        sub2sup = (x) -> Sub2SupItems[x]
        nsubitems = length(Sub2SupItems)
    catch
        sub2sup = (x) -> x
        nsubitems = nitems
    end

    xItemDofs = VariableTargetAdjacency(Int32)
    itemEG = EG[1]
    pattern::String = dofmap_patterns[1]
    iEG = 1
    k = 0
    itemdofs = zeros(Int,maxdofs4item)
    localnode = 0
    localface = 0
    localedge = 0
    newdof::Int = 0
    offset::Int = 0
    pos::Int = 0
    q::Int = 0
    for subitem = 1 : nsubitems
        itemEG = xItemGeometries[subitem]
        item = sub2sup(subitem)
        iEG = findfirst(isequal(itemEG), EG)
        pattern = dofmap_patterns[iEG]
        for c = 1 : ncomponents
            offset = (c-1)*offset4component
            for k = 1 : length(pattern)
                q = dofmap_quantifiers[iEG][k]
                if pattern[k] == 'N'
                    for n = 1 : nnodes_for_geometry(itemEG)
                        for m = 1 : q
                            pos += 1
                            itemdofs[pos] = xItemNodes[n,item] + offset + (m-1)*nnodes
                        end
                    end
                    offset += nnodes*q
                elseif pattern[k] == 'F'
                    for n = 1 : nfaces_for_geometry(itemEG)
                        for m = 1 : q
                            pos += 1
                            itemdofs[pos] = xItemFaces[n,item] + offset + (m-1)*nfaces
                        end
                    end
                    offset += nfaces*q
                elseif pattern[k] == 'E'
                    for n = 1 : nedges_for_geometry(itemEG)
                        for m = 1 : q
                            pos += 1
                            itemdofs[pos] = xItemEdges[n,item] + offset + (m-1)*nedges
                        end
                    end
                    offset += nedges*q
                elseif pattern[k] == 'I'
                    for m = 1 : q
                        pos += 1
                        itemdofs[pos] = item + offset
                        offset += nitems
                    end
                end
            end
        end
        offset = ncomponents*offset4component
        for k = 1 : length(pattern)
            q = dofmap_quantifiers[iEG][k]
            if pattern[k] == 'f'
                for n = 1 : nfaces_for_geometry(itemEG)
                    for m = 1 : q
                        pos += 1
                        itemdofs[pos] = xItemFaces[n,item] + offset + (m-1)*nfaces
                    end
                end
                offset += nfaces*q
            elseif pattern[k] == 'e'
                for n = 1 : nedges_for_geometry(itemEG)
                    for m = 1 : q
                        pos += 1
                        itemdofs[pos] = xItemEdges[n,item] + offset + (m-1)*nedges
                    end
                end
                offset += nedges*q
            elseif pattern[k] == 'i'
                for m = 1 : q
                    pos += 1
                    itemdofs[pos] = item + offset
                    offset += nitems
                end
            end
        end
        append!(xItemDofs,itemdofs[1:pos])
        pos = 0
    end
    # save dofmap
    FES.dofmaps[DM] = xItemDofs
end

function init_dofmap!(FES::FESpace, DM::Type{<:DofMap})
    init_dofmap_from_pattern!(FES, DM)
end


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
function point_evaluation!(Target::AbstractArray{<:Real,1}, FES::FESpace{FEType}, ::Type{AT_NODES}, exact_function!::Function; items = [], component_offset::Int = 0) where {FEType <: AbstractFiniteElement}
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
        exact_function!(result,x)
        for k = 1 : ncomponents
            Target[j+offset4component[k]] = result[k]
        end    
    end
end


# edge integral means
# used e.g. for interpolation into P2, P2B finite elements
function ensure_edge_moments!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, AT::Type{<:AbstractAssemblyType}, exact_function!::Function; items = [], bonus_quadorder::Int = 0) where {FEType <: AbstractFiniteElement}

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
    integrate!(edgemeans, FE.xgrid, AT, exact_function!, bonus_quadorder, ncomponents; items = items)
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




###########################
# Finite Element Subtypes #
###########################
#
# subtypes of the finite element supertypes above
# each defined in its own file
  
# Hdiv-conforming elements (only vector-valued)
# lowest order
include("fedefs/hdiv_rt0.jl");
include("fedefs/hdiv_bdm1.jl");
include("fedefs/hdiv_rt1.jl");

# H1 conforming elements (also Crouzeix-Raviart)
# lowest order
include("fedefs/h1_p1.jl");
include("fedefs/h1_mini.jl");
include("fedefs/h1nc_cr.jl");
include("fedefs/h1v_br.jl"); # Bernardi--Raugel (only vector-valued, with coefficients)
# second order
include("fedefs/h1_p2.jl");
include("fedefs/h1_p2b.jl");

# L2 conforming elements
include("fedefs/l2_p0.jl"); # currently masked as H1 element
include("fedefs/l2_p1.jl"); # currently masked as H1 element

# Hcurl-conforming elements
include("fedefs/hcurl_n0.jl");




# dummy function
get_polynomialorder(::Type{<:AbstractFiniteElement}, ::Type{<:Vertex0D}) = 0;
