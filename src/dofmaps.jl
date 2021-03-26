"""
$(TYPEDEF)

Dofmaps are stored as an ExtendableGrids.AbstractGridAdjacency in the finite element space and collect
information with respect to different AssemblyTypes. They are generated automatically on demand and the dofmaps
associated to each subtype can be accessed via FESpace[DofMap].
"""
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

Dofmap4AssemblyType(::Type{ON_CELLS}) = CellDofs
Dofmap4AssemblyType(::Type{<:ON_FACES}) = FaceDofs
Dofmap4AssemblyType(::Type{ON_BFACES}) = BFaceDofs
Dofmap4AssemblyType(::Type{<:ON_EDGES}) = EdgeDofs
Dofmap4AssemblyType(::Type{ON_BEDGES}) = BEdgeDofs

EffAT4AssemblyType(::Type{ON_CELLS},::Type{ON_CELLS}) = ON_CELLS
EffAT4AssemblyType(::Type{ON_CELLS},::Type{<:ON_FACES}) = ON_FACES
EffAT4AssemblyType(::Type{ON_CELLS},::Type{ON_BFACES}) = ON_BFACES
EffAT4AssemblyType(::Type{ON_CELLS},::Type{<:ON_EDGES}) = ON_EDGES
EffAT4AssemblyType(::Type{ON_CELLS},::Type{ON_BEDGES}) = ON_BEDGES

EffAT4AssemblyType(::Type{ON_FACES},::Type{ON_CELLS}) = nothing
EffAT4AssemblyType(::Type{ON_FACES},::Type{<:ON_FACES}) = ON_CELLS
EffAT4AssemblyType(::Type{ON_FACES},::Type{<:ON_EDGES}) = ON_FACES
EffAT4AssemblyType(::Type{ON_FACES},::Type{<:ON_BEDGES}) = ON_BFACES

EffAT4AssemblyType(::Type{ON_EDGES},::Type{ON_CELLS}) = nothing
EffAT4AssemblyType(::Type{ON_EDGES},::Type{<:ON_FACES}) = nothing
EffAT4AssemblyType(::Type{ON_EDGES},::Type{<:ON_EDGES}) = ON_CELLS



function Dofmap4AssemblyType(FES::FESpace, AT::Type{<:AbstractAssemblyType})
    return FES[Dofmap4AssemblyType(EffAT4AssemblyType(typeof(FES).parameters[2],AT))]
end


function init_dofmap_from_pattern!(FES::FESpace{FEType}, DM::Type{<:DofMap}) where {FEType <: AbstractFiniteElement}
    ## Beware: Automatic broken DofMap generation currently only reliable for CellDofs

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
    FES[DM] = xItemDofs
end


function init_broken_dofmap!(FES::FESpace{FEType}, DM::Union{Type{BFaceDofs},Type{FaceDofs}}) where {FEType <: AbstractFiniteElement}
    
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

    xFaceCells = FES.xgrid[FaceCells]
    xCellNodes = FES.xgrid[CellNodes]
    xCellDofs = FES[CellDofs]
    nnodes = size(xgrid[Coordinates],2)
    ncells = num_sources(xgrid[CellNodes])
    xFaceNodes = FES.xgrid[SuperItemNodes4DofMap(DM)]
    xFaceGeometries = xgrid[ItemGeometries4DofMap(DM)]
    xFaceDofs = VariableTargetAdjacency(Int32)
    if DM == BFaceDofs
        xRealFace = FES.xgrid[BFaces]
        nfaces = length(xRealFace)
    elseif DM == FaceDofs
        nfaces = num_sources(xFaceNodes)
        xRealFace = 1:nfaces
    end

    if need_faces
        xCellFaces = xgrid[CellFaces]
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
    local_nodes = zeros(Int, max_num_targets_per_source(xFaceNodes))
    local_face::Int = 0
    local_dofs = zeros(Int, max_num_targets_per_source(xCellDofs))
    local_facedofs = zeros(Int, 2*max_num_targets_per_source(xCellDofs))
    nldofs::Int = 0
    localoffset::Int = 0
    node::Int = 0
    face::Int = 0
    cell::Int = 0
    nfacenodes::Int = 0
    nfaceedges::Int = 0
    ncellnodes::Int = 0
    ncelledges::Int = 0
    ncellfaces::Int = 0
    pos::Int = 0
    for f = 1 : nfaces
        face = xRealFace[f]
        localoffset = 0
        for c = 1 : 2
            cell = xFaceCells[c,face]
            if cell > 0
                faceEG = xFaceGeometries[f]
                iEG = findfirst(isequal(faceEG), EG)
                pattern = dofmap_patterns[iEG]

                if need_nodes
                    ncellnodes = num_targets(xCellNodes,cell)
                    nfacenodes = num_targets(xFaceNodes,face)
                    for k = 1 : nfacenodes
                        node = xFaceNodes[k,face]
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
                    nfaceedges = num_targets(xFaceEdges,face)
                    for k = 1 : nfaceedges
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
                    for k = 1 : nfacenodes
                        for c = 1 : ncomponents
                            local_dofs[(c-1)*nfacenodes + k] = xCellDofs[1,cell] - 1 + (c-1)*ncellnodes + local_nodes[k]
                        end
                    end
                    nldofs = nfacenodes*ncomponents
                elseif pattern == "NC" # currently assumes quantifier = 1
                    for k = 1 : nfacenodes
                        for c = 1 : ncomponents
                            local_dofs[(c-1)*nfacenodes + k] = xCellDofs[1,cell] - 1 + (c-1)*(ncellnodes+1) + local_nodes[k]
                        end
                    end
                    nldofs = nfacenodes*ncomponents
                elseif pattern == "Ni" # currently assumes quantifier = 1
                    for k = 1 : nfacenodes
                        for c = 1 : ncomponents
                            local_dofs[(c-1)*nfacenodes + k] = xCellDofs[1,cell] - 1 + (c-1)*ncellnodes + local_nodes[k]
                        end
                    end
                    nldofs = nfacenodes*ncomponents + 1
                    local_dofs[nldofs] = xCellDofs[1,cell] - 1 + ncomponents*ncellnodes + local_face
                elseif pattern == "NI" # currently assumes quantifier = 1
                        for c = 1 : ncomponents
                            for k = 1 : nfacenodes
                                local_dofs[(c-1)*(nfacenodes+1) + k] = xCellDofs[1,cell] - 1 + (c-1)*(ncellnodes + ncellfaces) + local_nodes[k]
                            end
                            local_dofs[c*(nfacenodes+1)] = xCellDofs[1,cell] - 1 + c * ncellnodes + (c-1)*ncellfaces + local_face
                        end
                    nldofs = (nfacenodes+1)*ncomponents
                elseif pattern == "NE" # currently assumes quantifier = 1
                        for c = 1 : ncomponents
                            for k = 1 : nfacenodes
                                local_dofs[(c-1)*(nfacenodes+nfaceedges) + k] = xCellDofs[1,cell] - 1 + (c-1)*(ncellnodes + ncelledges) + local_nodes[k]
                            end
                            for k = 1 : nfaceedges
                                local_dofs[c*nfacenodes+(c-1)*nfaceedges + k] = xCellDofs[1,cell] - 1 + c * ncellnodes + (c-1)*ncelledges + local_edges[k]
                            end
                        end
                    nldofs = (nfacenodes+nfaceedges)*ncomponents
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
                    for k = 1 : nfaceedges
                        local_dofs[k] = xCellDofs[1,cell] - 1 + local_edges[k]
                    end
                    nldofs = nfaceedges
                elseif pattern == "E" # currently assumes quantifier = 1
                    for k = 1 : nfaceedges
                        for c = 1 : ncomponents
                            local_dofs[(c-1)*nfaceedges + k] = xCellDofs[1,cell] - 1 + (c-1)*ncelledges + local_edges[k]
                        end
                    end
                    nldofs = nfaceedges*ncomponents
                end
                for j = 1 : nldofs
                    local_facedofs[localoffset+j] = local_dofs[j]
                end
                localoffset += nldofs
            end
        end
        append!(xFaceDofs,local_facedofs[1:localoffset])
    end
    FES[DM] = xFaceDofs
end




function init_dofmap!(FES::FESpace, DM::Type{<:DofMap})
    @logmsg DeepInfo "Generating dofmap $DM for FESpace $(FES.name)"

    if (FES.broken == true) && (DM != CellDofs)
        init_broken_dofmap!(FES, DM)
    else
        init_dofmap_from_pattern!(FES, DM)
    end
end
