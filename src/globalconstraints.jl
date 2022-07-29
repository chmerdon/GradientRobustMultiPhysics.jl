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
    current_equations = "all")

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
    current_equations = "all") where {T,Tv,Ti}

    fixed_dofs = []

    c = Constraint.component
    c2 = Constraint.componentY
    dofsX = Constraint.dofsX
    dofsY = Constraint.dofsY
    factors::Array{T} = Constraint.factors
    @logmsg DeepInfo "Combining dofs of component $c and $c2..."
    
    AE::ExtendableSparseMatrix{Tv,Ti} = A.entries
    AM::SparseMatrixCSC{Tv,Ti} = A.entries.cscmatrix
    Avals::Array{Tv,1} = AM.nzval
    rows::Array{Ti,1} = rowvals(AM)
    targetrow::Int = 0
    sourcerow::Int = 0
    targetcol::Int = 0
    sourcecol::Int = 0
    diffY = A[c,c].offsetY - A[c2,c2].offsetY
    val::Float64 = 0
    for gdof = 1 : length(Constraint.dofsX)
        # copy source row (for dofY) to target row (for dofX)
        targetrow = dofsX[gdof] + A[c,c].offsetX
        sourcerow = A[c2,c2].offsetX + dofsY[gdof]
        for col = 1 : size(A[c2,c2],2)
            sourcecol = col + A[c2,c2].offsetY
            targetcol = sourcecol + diffY
            val = AE[sourcerow,sourcecol]
            if abs(val) > 1e-14
                _addnz(AE, targetrow,targetcol, factors[gdof] * val,1)
                AE[sourcerow,sourcecol] = 0
            end
        end

        # replace source row (of dofY) with equation for coupling the two dofs
        sourcecol = A[c2,c2].offsetY + dofsY[gdof]
        targetcol = dofsX[gdof] + A[c,c].offsetY
        targetrow = dofsX[gdof] + A[c,c].offsetX
        sourcerow = A[c2,c2].offsetX + dofsY[gdof]
        _addnz(AE, sourcerow, targetcol, 1,1)
        _addnz(AE, sourcerow, sourcecol, -factors[gdof],1)
    end

    # fix one of the dofs
    for gdof = 1 : length(Constraint.dofsX)
        targetrow = b[c].offset + dofsX[gdof]
        sourcerow = b[c2].offset + dofsY[gdof]
        b.entries[targetrow] += b.entries[sourcerow]
        #Target.entries[sourcerow] = 0
        b.entries[sourcerow] = 0
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
    current_equations = "all")

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
