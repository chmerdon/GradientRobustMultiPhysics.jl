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

get_edim(FEType::Type{<:AbstractFiniteElement}) = 0 # not defined


"""
$(TYPEDEF)

A struct that has a finite element type as parameter and carries dofmaps (CellDofs, FaceDofs, BFaceDofs) plus additional grid information and access to arrays holding coefficients if needed.
"""
mutable struct FESpace{FEType<:AbstractFiniteElement}
    name::String                          # full name of finite element space (used in messages)
    broken::Bool                          # if true, broken dofmaps are generated
    ndofs::Int                            # total number of dofs
    xgrid::ExtendableGrid                 # link to xgrid 
    dofmaps::Dict{Type{<:AbstractGridComponent},Any} # backpack with dofmaps
end

function FESpace{FEType}(xgrid::ExtendableGrid; name = "", dofmaps_needed = "auto", broken::Bool = false, verbosity = 0 ) where {FEType <:AbstractFiniteElement}
    # piecewise constants are always broken
    if FEType <: H1P0 
        broken = true
    end
    
    # first generate some empty FESpace
    dummyVTA = VariableTargetAdjacency(Int32)
    FES = FESpace{FEType}(name,broken,0,xgrid,Dict{Type{<:AbstractGridComponent},Any}())

    if verbosity > 0
        println("  Initialising FESpace $FEType...")
    end
    FES.name = broken ? "$FEType (broken)" : "$FEType"

    if dofmaps_needed == "auto"
        if broken == true
            dofmaps_needed = [CellDofs, BFaceDofs]
        else
            dofmaps_needed = [CellDofs, FaceDofs, BFaceDofs]
        end
        if FEType <: AbstractHcurlFiniteElement && get_ncomponents(FEType) == 3
            if broken == false
                push!(dofmaps_needed, EdgeDofs)
            end
            # push!(dofmaps_needed, BEdgeDofs) # 3D Hcurl boundary data not working properly yet
        end
        edim = get_edim(FEType)
        if FEType <: H1P2 && edim == 3
            if broken == false
                push!(dofmaps_needed, EdgeDofs)
            end
        end
    else
        if broken == true
            @assert dofmaps_needed[1] == CellDofs
        end
    end

    count_ndofs!(FES)

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
    ##

    if FES.broken == true && !(DM in [CellDofs])
        println("Automatic broken DofMap generation currently only available for CellDofs (but requested $DM)")
        return nothing
    end

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
            elseif dofmap_patterns[j][k] == 'e'
                dofs4item_single += nedges_for_geometry(EG[j])*dofmap_quantifiers[j][k]
                need_edges = true
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

    if FES.broken == true
        xItemDofs = SerialVariableTargetAdjacency(Int32)
    else
        xItemDofs = VariableTargetAdjacency(Int32)
    end
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
    total_broken_dofs::Int = 0
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
        if FES.broken
            append!(xItemDofs,pos)
        else
            append!(xItemDofs,itemdofs[1:pos])
        end
        pos = 0
    end
    # save dofmap
    FES.dofmaps[DM] = xItemDofs
end


function init_broken_dofmap!(FES::FESpace{FEType}, DM::Type{BFaceDofs}) where {FEType <: AbstractFiniteElement}
    
    ## prepare dofmap patterns
    xgrid = FES.xgrid
    EG = xgrid[UCG4DofMap(DM)]
    ncomponents::Int = get_ncomponents(FEType)
    need_nodes = false
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
                need_nodes = true
            #elseif dofmap_patterns[j][k] == 'F'
            #    dofs4item4component += nfaces_for_geometry(EG[j])*dofmap_quantifiers[j][k]
            #    need_faces = true
            elseif dofmap_patterns[j][k] == 'E'
                dofs4item4component += nedges_for_geometry(EG[j])*dofmap_quantifiers[j][k]
                need_edges = true
            elseif dofmap_patterns[j][k] in ['I','C']
                dofs4item4component += dofmap_quantifiers[j][k]
                need_faces = true
            #elseif dofmap_patterns[j][k] == 'f'
            #    dofs4item_single += nfaces_for_geometry(EG[j])*dofmap_quantifiers[j][k]
            #    need_faces = true
            elseif dofmap_patterns[j][k] == 'e'
                dofs4item_single += nedges_for_geometry(EG[j])*dofmap_quantifiers[j][k]
                need_edges = true
            elseif dofmap_patterns[j][k] in ['i','c'] 
                dofs4item_single += dofmap_quantifiers[j][k]
                need_faces = true
            end
        end
        maxdofs4item = max(maxdofs4item,dofs4item4component*ncomponents + dofs4item_single)
        dofs4item4component = 0
        dofs4item_single = 0
    end

    xBFaceNodes = FES.xgrid[BFaceNodes]
    nbfaces = num_sources(xBFaceNodes)
    xBFaces = FES.xgrid[BFaces]
    xFaceCells = FES.xgrid[FaceCells]
    xCellNodes = FES.xgrid[CellNodes]
    xCellDofs = FES.dofmaps[CellDofs]
    xBFaceDofs = VariableTargetAdjacency(Int32)
    nnodes = size(xgrid[Coordinates],2)
    ncells = num_sources(xgrid[CellNodes])
    xBFaceGeometries = xgrid[ItemGeometries4DofMap(DM)]
    if need_faces
        xCellFaces = xgrid[CellFaces]
        nfaces = num_sources(xgrid[FaceNodes])
    end
    if need_edges
        xFaceEdges = xgrid[FaceEdges]
        xCellEdges = xgrid[CellEdges]
        nedges = num_sources(xgrid[EdgeNodes])
        local_edges = zeros(Int, max_num_targets_per_source(xFaceEdges))
    end

    itemEG = EG[1]
    pattern::String = dofmap_patterns[1]
    iEG::Int = 1
    itemdofs = zeros(Int,maxdofs4item)
    local_nodes = zeros(Int, max_num_targets_per_source(xBFaceNodes))
    local_face::Int = 0
    local_dofs = zeros(Int, max_num_targets_per_source(xCellDofs))
    nldofs::Int = 0
    node::Int = 0
    face::Int = 0
    cell::Int = 0
    nbfacenodes::Int = 0
    nbfaceedges::Int = 0
    ncellnodes::Int = 0
    ncelledges::Int = 0
    ncellfaces::Int = 0
    pos::Int = 0
    for bface = 1 : nbfaces
        face = xBFaces[bface]
        cell = xFaceCells[1,face]
        bfaceEG = xBFaceGeometries[bface]
        iEG = findfirst(isequal(bfaceEG), EG)
        pattern = dofmap_patterns[iEG]

        if need_nodes
            ncellnodes = num_targets(xCellNodes,cell)
            nbfacenodes = num_targets(xBFaceNodes,bface)
            for k = 1 : nbfacenodes
                node = xBFaceNodes[k,bface]
                pos = 1
                while xCellNodes[pos,cell] != node
                    pos += 1
                end
                local_nodes[k] = pos
            end
        end
        if need_faces
            ncellfaces = num_targets(xCellFaces,cell)
            pos = 1
            while xCellFaces[pos,cell] != face
                pos += 1
            end
            local_face = pos
        end
        if need_edges
            ncelledges = num_targets(xCellEdges,cell)
            nbfaceedges = num_targets(xFaceEdges,face)
            for k = 1 : nbfaceedges
                edge = xFaceEdges[k, face]
                pos = 1
                while xCellEdges[pos,cell] != edge
                    pos += 1
                end
                local_edges[k] = pos
            end
        end

        if pattern == "C" # currently assumes quantifier = 1
            for c = 1 : ncomponents
                local_dofs[c] = xCellDofs[1,cell] + c - 1
            end
            nldofs = ncomponents
        elseif pattern == "N" # currently assumes quantifier = 1
            for k = 1 : nbfacenodes
                for c = 1 : ncomponents
                    local_dofs[(c-1)*nbfacenodes + k] = xCellDofs[1,cell] - 1 + (c-1)*ncellnodes + local_nodes[k]
                end
            end
            nldofs = nbfacenodes*ncomponents
        elseif pattern == "NC" # currently assumes quantifier = 1
            for k = 1 : nbfacenodes
                for c = 1 : ncomponents
                    local_dofs[(c-1)*nbfacenodes + k] = xCellDofs[1,cell] - 1 + (c-1)*(ncellnodes+1) + local_nodes[k]
                end
            end
            nldofs = nbfacenodes*ncomponents
        elseif pattern == "Ni" # currently assumes quantifier = 1
            for k = 1 : nbfacenodes
                for c = 1 : ncomponents
                    local_dofs[(c-1)*nbfacenodes + k] = xCellDofs[1,cell] - 1 + (c-1)*ncellnodes + local_nodes[k]
                end
            end
            nldofs = nbfacenodes*ncomponents + 1
            local_dofs[nldofs] = xCellDofs[1,cell] - 1 + ncomponents*ncellnodes + local_face
        elseif pattern == "NI" # currently assumes quantifier = 1
                for c = 1 : ncomponents
                    for k = 1 : nbfacenodes
                        local_dofs[(c-1)*(nbfacenodes+1) + k] = xCellDofs[1,cell] - 1 + (c-1)*(ncellnodes + ncellfaces) + local_nodes[k]
                    end
                    local_dofs[c*(nbfacenodes+1)] = xCellDofs[1,cell] - 1 + c * ncellnodes + (c-1)*ncellfaces + local_face
                end
            nldofs = (nbfacenodes+1)*ncomponents
        elseif pattern == "NE" # currently assumes quantifier = 1
                for c = 1 : ncomponents
                    for k = 1 : nbfacenodes
                        local_dofs[(c-1)*(nbfacenodes+nbfaceedges) + k] = xCellDofs[1,cell] - 1 + (c-1)*(ncellnodes + ncelledges) + local_nodes[k]
                    end
                    for k = 1 : nbfaceedges
                        local_dofs[c*nbfacenodes+(c-1)*nbfaceedges + k] = xCellDofs[1,cell] - 1 + c * ncellnodes + (c-1)*ncelledges + local_edges[k]
                    end
                end
            nldofs = (nbfacenodes+nbfaceedges)*ncomponents
        elseif pattern == "i"
            quantifier = dofmap_quantifiers[iEG][1]
            for q = 1 : quantifier
                local_dofs[q] = xCellDofs[1,cell] - 1 + (local_face-1)*quantifier + q
            end
            nldofs = quantifier
        elseif pattern == "I" # currently assumes quantifier = 1
            for c = 1 : ncomponents
                local_dofs[c] = xCellDofs[1,cell] - 1 + (c-1)*ncellfaces + local_face
            end
            nldofs = ncomponents
        elseif pattern == "e" # currently assumes quantifier = 1
            for k = 1 : nbfaceedges
                local_dofs[k] = xCellDofs[1,cell] - 1 + local_edges[k]
            end
            nldofs = nbfaceedges
        elseif pattern == "E" # currently assumes quantifier = 1
            for k = 1 : nbfaceedges
                for c = 1 : ncomponents
                    local_dofs[(c-1)*nbfaceedges + k] = xCellDofs[1,cell] - 1 + (c-1)*ncelledges + local_edges[k]
                end
            end
            nldofs = nbfaceedges*ncomponents
        end
        append!(xBFaceDofs,local_dofs[1:nldofs])
    end
    FES.dofmaps[BFaceDofs] = xBFaceDofs
end

function init_dofmap!(FES::FESpace, DM::Type{<:DofMap})
    if (FES.broken == true) && (DM != CellDofs)
        init_broken_dofmap!(FES, DM)
    else
        init_dofmap_from_pattern!(FES, DM)
    end
end

function count_ndofs!(FES::FESpace{FEType}) where {FEType <: AbstractFiniteElement}
    xgrid = FES.xgrid
    EG = xgrid[UniqueCellGeometries]
    xCellGeometries = xgrid[CellGeometries]
    ncells4EG::Array{Tuple{DataType,Int64},1} = [(i, count(==(i), xCellGeometries)) for i in EG]
    ndofs4EG = zeros(Int, length(EG))
    ncomponents::Int = get_ncomponents(FEType)
    dofs4item4component = zeros(Int,4)
    dofs4item_single = zeros(Int,4)
    maxdofs4item4component = zeros(Int,4)
    maxdofs4item_single = zeros(Int,4)
    quantifier::Int = 0
    pattern_char::Char = ' '
    highest_quantifierNCEF = zeros(Int,0)
    totaldofs = 0
    for j = 1 : length(EG)
        pattern = get_dofmap_pattern(FEType, CellDofs, EG[j])
        for k = 1 : Int(length(pattern)/2)
            pattern_char = pattern[2*k-1]
            quantifier = parse(Int,pattern[2*k])
            if pattern_char == 'N'
                dofs4item4component[1] += quantifier
            elseif pattern_char == 'F'
                dofs4item4component[2] += quantifier
            elseif pattern_char == 'E'
                dofs4item4component[3] += quantifier
            elseif pattern_char == 'I'
                dofs4item4component[4] += quantifier
            elseif pattern_char == 'n'
                dofs4item_single[1] += quantifier
            elseif pattern_char == 'f'
                dofs4item_single[2] += quantifier
            elseif pattern_char == 'e'
                dofs4item_single[3] += quantifier
            elseif pattern_char == 'i'
                dofs4item_single[4] += quantifier
            end
        end
        ndofs4EG[j] = nnodes_for_geometry(EG[j]) * (ncomponents * dofs4item4component[1] + dofs4item_single[1])
        ndofs4EG[j] += nfaces_for_geometry(EG[j]) * (ncomponents * dofs4item4component[2] + dofs4item_single[2])
        ndofs4EG[j] += nedges_for_geometry(EG[j]) * (ncomponents * dofs4item4component[3] + dofs4item_single[3])
        ndofs4EG[j] += ncomponents * dofs4item4component[4] + dofs4item_single[4]
        if FES.broken == true
            totaldofs += ndofs4EG[j] * ncells4EG[j][2]
        else
            # only count interior dofs on cell here
            totaldofs +=  ncells4EG[j][2] * (ncomponents * dofs4item4component[4] + dofs4item_single[4])
            for n = 1 : 4
                maxdofs4item4component[n] = max(maxdofs4item4component[n], dofs4item4component[n])
                maxdofs4item_single[n] = max(maxdofs4item_single[n], dofs4item_single[n])
            end
        end
        fill!(dofs4item4component,0)
        fill!(dofs4item_single,0)
    end
    if FES.broken == false
        # add continuous dofs here
        if maxdofs4item4component[1] + maxdofs4item_single[1] > 0
            totaldofs += size(xgrid[Coordinates],2) * (ncomponents * maxdofs4item4component[1] + maxdofs4item_single[1])
        end
        if maxdofs4item4component[2] + maxdofs4item_single[2] > 0
            totaldofs += num_sources(xgrid[FaceNodes]) * (ncomponents * maxdofs4item4component[2] + maxdofs4item_single[2])
        end
        if maxdofs4item4component[3] + maxdofs4item_single[3] > 0
            totaldofs += num_sources(xgrid[EdgeNodes]) * (ncomponents * maxdofs4item4component[3] + maxdofs4item_single[3])
        end
    end
    FES.ndofs = totaldofs
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
    xCellDofs = FES.dofmaps[CellDofs]

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
include("fedefs/h1_p0.jl");
include("fedefs/h1_p1.jl");
include("fedefs/h1_mini.jl");
include("fedefs/h1nc_cr.jl");
include("fedefs/h1v_br.jl"); # Bernardi--Raugel (only vector-valued, with coefficients)
# second order
include("fedefs/h1_p2.jl");
include("fedefs/h1_p2b.jl");

# Hcurl-conforming elements
include("fedefs/hcurl_n0.jl");




# dummy function
get_polynomialorder(::Type{<:AbstractFiniteElement}, ::Type{<:Vertex0D}) = 0;
