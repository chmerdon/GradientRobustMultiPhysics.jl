#################################
### AbstractGlobalConstraints ###
#################################
#
# further constraints that cannot be described with (sparse) PDEOperators and are realized by manipulations of the
# already assembled system
#
# FixedIntegralMean: Ensure that integral mean of a _component_ attains _value_
# (e.g. for pressure in Stokes, avoids full row/column in matrix)
#
# CombineDofs: Identify given dofs of two components with each other
# (might be used for periodic boundary conditions, or using different FETypes in different regions)
#
# NOTE: constraints like zero divergence have to be realised with PDEOperators like LagrangeMultiplier
#
abstract type AbstractGlobalConstraint end

"""
$(TYPEDEF)

fixes integral mean of the unknown to the specified value
"""
struct FixedIntegralMean <: AbstractGlobalConstraint
    name::String
    component::Int
    value::Real
    when_assemble::Type{<:AbstractAssemblyTrigger}
end 


"""
````
function FixedIntegralMean(unknown_id::Int, value::Real; name::String = "")
````

constructs a FixedIntegralMean constraint that (if assigned to a PDEDescription) ensures that the unknown with the specified id has an integral mean value.

"""
function FixedIntegralMean(component::Int, value::Real; name::String = "")
    if name == ""
        name = "Mean[$component] != $value"
    end
    return FixedIntegralMean(name,component, value, AssemblyFinal)
end

"""
$(TYPEDEF)

combines specified degrees of freedom of two unknown (can be the same), which allows to glue together different unknowns in different regions or periodic boundary conditions
"""
struct CombineDofs{T} <: AbstractGlobalConstraint
    name::String
    component::Int                  # component nr for dofsX
    componentY::Int                  # component nr for dofsY
    dofsX::Array{Int,1}     # dofsX that should be the same as dofsY in Y component
    dofsY::Array{Int,1}
    factors::AbstractArray{T,1}
    when_assemble::Type{<:AbstractAssemblyTrigger}
end 



"""
````
function get_periodic_coupling_info(FES, xgrid, b1, b2, is_opposite::Function; factor_vectordofs = "auto")
````

computes the dofs that have to be coupled for periodic boundary conditions on the given xgrid for boundary regions b1, b2.
The is_opposite function evaluates if two provided face midpoints are on opposite sides to each other (the mesh xgrid should be appropriate).
For vector-valued FETypes the user can provide factor_vectordofs to incorporate a sign change if needed.
This is automatically done for all Hdiv-conforming elements and (for the normal-weighted face bubbles of) the Bernardi-Raugel element H1BR. 

"""
function get_periodic_coupling_info(
    FES::FESpace,
    xgrid::ExtendableGrid,
    b1,
    b2,
    is_opposite::Function;
    factor_vectordofs = "auto",
    factor_components = "auto")

    FEType = eltype(FES)
    ncomponents = get_ncomponents(FEType)
    if factor_vectordofs == "auto"
        if FEType <: AbstractHdivFiniteElement || FEType <: H1BR
            factor_vectordofs = -1
        else
            factor_vectordofs = 1
        end
    end
    if factor_components == "auto"
        factor_components = ones(Int, ncomponents)
    end


    @assert FEType <: AbstractH1FiniteElement "not yet working for non H1-conforming elements"
    xBFaceRegions = xgrid[BFaceRegions]
    xBFaceNodes = xgrid[BFaceNodes]
    xBFaceFaces = xgrid[BFaceFaces]
    xCoordinates = xgrid[Coordinates]
    nbfaces = size(xBFaceNodes,2)
    nnodes = num_nodes(xgrid)
    nnodes4bface = size(xBFaceNodes,1)
    EG = xgrid[UniqueBFaceGeometries][1]
    xdim = size(xCoordinates,1)
    nedges4bface = xdim == 3 ? num_faces(EG) : 0
    xBFaceMidPoints = zeros(Float64,xdim,nbfaces)
    for bface = 1 : nbfaces, j = 1 : xdim, bn = 1 : nnodes4bface 
        xBFaceMidPoints[j,bface] += xCoordinates[j,xBFaceNodes[bn,bface]] / nnodes4bface 
    end
    if xdim == 3
        xEdgeMidPoint = zeros(Float64,xdim)
        xEdgeMidPoint2 = zeros(Float64,xdim)
        xEdgeNodes = xgrid[EdgeNodes]
        xFaceEdges = xgrid[FaceEdges]
        if FEType <: H1P1
            nedgedofs = 0
        elseif FEType <: H1P2
            nedgedofs = 1
        elseif FEType <: H1P3
            nedgedofs = 2
        else
            @warn "get_periodic_coupling_info not yet working for non H1-conforming elements"
        end
    end
    @assert FEType <: AbstractH1FiniteElement "get_periodic_coupling_info not yet working for non H1-conforming elements"
    xBFaceDofs = FES[BFaceDofs]
    dofsX, dofsY, factors = Int[], Int[], Int[]
    counterface = 0
    nfb = 0
    partners = zeros(Int, xdim)
    coffsets = get_local_coffsets(FEType, ON_BFACES, EG)
    nedgedofs = 0
    
    for bface = 1 : nbfaces
        counterface = 0
        nfb = num_targets(xBFaceDofs, bface)
        if xBFaceRegions[bface] == b1
            for bface2 = 1 : nbfaces
                if xBFaceRegions[bface2] == b2
                    if is_opposite(view(xBFaceMidPoints,:,bface), view(xBFaceMidPoints,:,bface2))
                        counterface = bface2
                        break
                    end
                end
            end
        end
        if counterface > 0

            # couple first two node dofs in opposite order due to orientation
            for c = 1 : ncomponents
                if factor_components[c] == 0
                    continue
                end
                nfbc = coffsets[c+1] - coffsets[c] # total dof count for this component

                # couple nodes
                for nb = 1 : nnodes4bface 
                    ## find node partner on other side that evaluates true in is_ooposite function
                    for nc = 1 : nnodes4bface 
                        if is_opposite(view(xCoordinates,:,xBFaceDofs[nb,bface]), view(xCoordinates,:,xBFaceDofs[nc,counterface]))
                            partners[nb] = nc
                            break
                        end 
                    end
                    ## couple node dofs (to be skipped for e.g. Hdiv, Hcurl elements)
                    push!(dofsX, xBFaceDofs[coffsets[c]+nb,bface])
                    push!(dofsY, xBFaceDofs[coffsets[c]+partners[nb],counterface])
                end
                # @info "matching face $bface (nodes = $(xBFaceNodes[:,bface]), dofs = $(xBFaceDofs[:,bface])) with face $counterface (nodes = $(xBFaceNodes[:,counterface]), dofs = $(xBFaceDofs[:,counterface])) with partner node order $partners"
            
                ## couple edges
                if nedges4bface > 0 && FEType <: H1P2 || FEType <: H1P3
                    # todo: for H1P3 edge orientation place a role !!!
                    for nb = 1 : nedges4bface 
                        fill!(xEdgeMidPoint,0)
                        for j = 1 : xdim, k = 1 : 2
                            xEdgeMidPoint[j] += xCoordinates[j,xEdgeNodes[k,xFaceEdges[nb,xBFaceFaces[bface]]]] / 2 
                        end
                        ## find edge partner on other side that evaluates true at edge midpoint in is_opposite function
                        for nc = 1 : nnodes4bface 
                            fill!(xEdgeMidPoint2,0)
                            for j = 1 : xdim, k = 1 : 2
                                xEdgeMidPoint2[j] += xCoordinates[j,xEdgeNodes[k,xFaceEdges[nc,xBFaceFaces[counterface]]]] / 2
                            end
                            if is_opposite(xEdgeMidPoint, xEdgeMidPoint2)
                                partners[nb] = nc
                                break
                            end 
                        end

                        ## couple edge dofs (local orientation information is needed for more than one dof on each edge !!! )
                        for k = 1 : nedgedofs
                            push!(dofsX, xBFaceDofs[coffsets[c]+nnodes4bface+nb+(k-1)*nedgedofs,bface])
                            push!(dofsY, xBFaceDofs[coffsets[c]+nnodes4bface+partners[nb]+(k-1)*nedgedofs,counterface])
                        end
                    end
                end

                ## couple face dofs (interior dofs of bface)
                for nb = 1 : nfbc-nnodes4bface-nedges4bface*nedgedofs
                    push!(dofsX, xBFaceDofs[coffsets[c]+nnodes4bface+nedges4bface*nedgedofs+nb,bface])
                    push!(dofsY, xBFaceDofs[coffsets[c]+nnodes4bface+nfbc-nnodes4bface-nedges4bface*nedgedofs + 1 - nb,counterface]) # couple face dofs in opposite order due to orientation (works in 2D at least)
                end
                append!(factors, ones(nfbc) * factor_components[c])
            end
            
            ## couple remaining dofs (should be vector dofs)
            for dof = coffsets[end]+1:nfb
                push!(dofsX, xBFaceDofs[dof,bface])
                push!(dofsY, xBFaceDofs[nfb-coffsets[end]+dof-1,counterface]) # couple face dofs in opposite order due to orientation (works in 2D at least, e.g. for Bernardi--Raugel)
                push!(factors, factor_vectordofs)
            end
        end
    end

    return dofsX, dofsY, factors
end


"""
````
function CombineDofs(idX::Int,idY::Int,dofsX::Array{Int,1},dofsY::Array{Int,1})
````

constructs a CombineDofs constraint that (if assigned to a PDEDescription) ensures that the dofsX of the unknown with id idX matches the dofsY of the unknown with id idY.

"""
function CombineDofs(componentX::Int,componentY::Int,dofsX::Array{Int,1},dofsY::Array{Int,1},factors = ones(Int,length(dofsX)))
    @assert length(dofsX) == length(dofsY)
    return CombineDofs{eltype(factors)}("CombineDofs[$componentX,$componentY] (ndofs = $(length(dofsX)))",componentX,componentY,dofsX,dofsY,factors, AssemblyAlways)
end

function apply_constraint!(
    A::FEMatrix,
    b::FEVector,
    Constraint::FixedIntegralMean,
    Target::FEVector;
    current_equations = "all",
    lhs_mask = nothing,
    rhs_mask = nothing)

    c = Constraint.component
    @logmsg DeepInfo "Ensuring fixed integral mean for component $c..."

    if current_equations != "all"
        c = findfirst(isequal(c), current_equations)
    end

    # chose the the first dof in block coresspnding to component c
    # that will be fixed (and later moved such that desired mean is reached)
    dof = A[c,c].offsetX+1
    #Target[c][1] = 0
    return [dof]
end


function apply_constraint!(
    A::FEMatrix{Tv,Ti},
    b::FEVector,
    Constraint::CombineDofs{T},
    Target::FEVector;
    lhs_mask = nothing,
    rhs_mask = nothing,
    current_equations = "all") where {T,Tv,Ti}

    fixed_dofs = []


    c = Constraint.component
    c2 = Constraint.componentY

    if current_equations != "all"
        c = findfirst(isequal(c), current_equations)
        c2 = findfirst(isequal(c2), current_equations)
    end

    dofsX = Constraint.dofsX
    dofsY = Constraint.dofsY
    factors::Array{T} = Constraint.factors
    @logmsg DeepInfo "Combining dofs of component $c and $c2..."
    
    AE::ExtendableSparseMatrix{Tv,Ti} = A.entries
    targetrow::Int = 0
    sourcerow::Int = 0
    targetcol::Int = 0
    sourcecol::Int = 0
    val::Float64 = 0
    for gdof = 1 : length(Constraint.dofsX)
        # copy source row (for dofY) to target row (for dofX)
        targetrow = dofsX[gdof] + A[c,c].offsetX
        sourcerow = A[c2,c2].offsetX + dofsY[gdof]
        for b = 1 : nbcols(A)
            cblock = A[c,b]
            if lhs_mask === nothing || lhs_mask[c,b]
                for sourcecol = cblock.offsetY+1 : cblock.last_indexY
                    targetcol = sourcecol - A[c2,c2].offsetY + A[c,c].offsetY
                    val = AE[sourcerow,sourcecol]
                    if abs(val) > 1e-14
                        _addnz(AE, targetrow, targetcol, factors[gdof] * val,1)
                        AE[sourcerow,sourcecol] = 0
                    end
                end
            end
        end

        # replace source row (of dofY) with equation for coupling the two dofs
        sourcecol = dofsY[gdof] + A[c2,c2].offsetY
        targetcol = dofsX[gdof] + A[c,c].offsetY
        sourcerow = A[c2,c2].offsetX + dofsY[gdof]
        if lhs_mask === nothing || lhs_mask[c2,c] 
            _addnz(AE, sourcerow, targetcol, 1,1)
        end
        if lhs_mask === nothing || lhs_mask[c2,c2]
            _addnz(AE, sourcerow, sourcecol, -factors[gdof],1)
        end
    end

    # fix one of the dofs
    for gdof = 1 : length(Constraint.dofsX)
        sourcerow = b[c2].offset + dofsY[gdof]
        targetrow = b[c].offset + dofsX[gdof]
        if rhs_mask === nothing || rhs_mask[c]
            b.entries[targetrow] += b.entries[sourcerow]
        end
        if rhs_mask === nothing || rhs_mask[c2]
            b.entries[sourcerow] = 0
        end
        #Target.entries[sourcerow] = 0
        #push!(fixed_dofs,sourcerow)
    end
    flush!(A.entries)
    return fixed_dofs
end

function realize_constraint!(
    Target::FEVector{T,Tv,Ti},
    Constraint::FixedIntegralMean) where {T,Tv,Ti}

    c = Constraint.component
    
    @logmsg MoreInfo "Moving integral mean for component $c to value $(Constraint.value)"

    # move integral mean
    pmeanIntegrator = ItemIntegrator([Identity])
    total_area = sum(Target.FEVectorBlocks[c].FES.xgrid[CellVolumes], dims=1)[1]
    meanvalue = evaluate(pmeanIntegrator,Target[c])/total_area
    for dof=1:Target.FEVectorBlocks[c].FES.ndofs
        Target[c][dof] += Constraint.value - meanvalue
    end    
end


function realize_constraint!(
    Target::FEVector,
    Constraint::CombineDofs)

    c = Constraint.component
    c2 = Constraint.componentY
    @debug "Moving entries of combined dofs from component $c to component $c2"

    for dof = 1 : length(Constraint.dofsX)
        #Target[c2][Constraint.dofsY[dof]] = Target[c][Constraint.dofsX[dof]]
        #Target[c][Constraint.dofsX[dof]] = Target[c2][Constraint.dofsY[dof]]
    end 
end



"""
$(TYPEDEF)

fixes integral mean of the unknown to the specified value
"""
struct FixedDofs{T} <: AbstractGlobalConstraint
    name::String
    component::Int
    dofs::AbstractArray{Int}
    values::AbstractArray{T}
    when_assemble::Type{<:AbstractAssemblyTrigger}
end 


"""
````
function FixedDofs(unknown_id::Int, value::Real; name::String = "")
````

constructs a FixedDofs constraint that (if assigned to a PDEDescription) ensures that the dofs are fixed to the specified values.

"""
function FixedDofs(component::Int, dofs, values; name::String = "")
    if name == ""
        name = "FixedDofs[$component] ($(length(dofs)) many)"
    end
    return FixedDofs{eltype(values)}(name, component, dofs, values, AssemblyAlways)
end


function apply_constraint!(
    A::FEMatrix,
    b::FEVector,
    Constraint::FixedDofs,
    Target::FEVector;
    current_equations = "all",
    lhs_mask = nothing,
    rhs_mask = nothing)

    c = Constraint.component
    @logmsg DeepInfo "Ensuring fixed dofs for component $c..."

    if current_equations != "all"
        c = findfirst(isequal(c), current_equations)
    end

    dofs = Constraint.dofs
    values = Constraint.values
    for j = 1 : length(dofs)
        Target[c][dofs[j]] = values[j]
    end
    return Constraint.dofs
end

function realize_constraint!(
    Target::FEVector{T,Tv,Ti},
    Constraint::FixedDofs) where {T,Tv,Ti}
    
    # do nothing
end
