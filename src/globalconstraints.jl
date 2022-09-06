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
function get_periodic_coupling_info(FES, xgrid, b1, b2, is_opposite::Function; factor_vectordofs = "auto")
    if factor_vectordofs == "auto"
        if eltype(FES) <: AbstractHdivFiniteElement || eltype(FES) <: H1BR
            factor_vectordofs = -1
        end
    end
    xBFaceRegions = xgrid[BFaceRegions]
    xBFaceNodes = xgrid[BFaceNodes]
    xCoordinates = xgrid[Coordinates]
    nbfaces = size(xBFaceNodes,2)
    xdim = size(xCoordinates,1)
    xBFaceMidPoints = zeros(Float64,xdim,nbfaces)
    for bface = 1 : nbfaces, j = 1 : xdim, bn = 1 : 2
        xBFaceMidPoints[j,bface] += xCoordinates[j,xBFaceNodes[bn,bface]] / xdim
    end
    xBFaceDofs = FES[BFaceDofs]
    dofsX, dofsY, factors = Int[], Int[], Int[]
    counterface = 0
    nfb = 0
    ncomponents = get_ncomponents(eltype(FES))
    coffsets = get_local_coffsets(eltype(FES), ON_BFACES, Edge1D)
    for bface = 1 : nbfaces
        counterface = 0
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
            nfb = num_targets(xBFaceDofs, bface)
            # couple first two node dofs in opposite order due to orientation
            for c = 1 : ncomponents
                push!(dofsX, xBFaceDofs[coffsets[c]+1,bface])
                push!(dofsY, xBFaceDofs[coffsets[c]+2,counterface]) 
                push!(dofsX, xBFaceDofs[coffsets[c]+2,bface])
                push!(dofsY, xBFaceDofs[coffsets[c]+1,counterface])
                nfbc = coffsets[c+1] - coffsets[c]
                for dof = 1 : nfbc-2
                    push!(dofsX, xBFaceDofs[coffsets[c]+2+dof,bface])
                    push!(dofsY, xBFaceDofs[coffsets[c]+1+nfbc-dof,counterface]) # couple face dofs in opposite order due to orientation
                end
                append!(factors, ones(nfbc))
            end
            for dof = coffsets[end]+1:nfb
                push!(dofsX, xBFaceDofs[dof,bface])
                push!(dofsY, xBFaceDofs[nfb-coffsets[end]+dof-1,counterface]) # couple face dofs in opposite order due to orientation#
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
    lhs_erased = nothing,
    rhs_erased = nothing)

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
