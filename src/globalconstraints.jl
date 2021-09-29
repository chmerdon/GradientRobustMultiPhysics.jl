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
struct CombineDofs <: AbstractGlobalConstraint
    name::String
    componentX::Int                  # component nr for dofsX
    componentY::Int                  # component nr for dofsY
    dofsX::Array{Int,1}     # dofsX that should be the same as dofsY in Y component
    dofsY::Array{Int,1}
    when_assemble::Type{<:AbstractAssemblyTrigger}
end 

"""
````
function CombineDofs(idX::Int,idY::Int,dofsX::Array{Int,1},dofsY::Array{Int,1})
````

constructs a CombineDofs constraint that (if assigned to a PDEDescription) ensures that the dofsX of the unknown with id idX matches the dofsY of the unknown with id idY.

"""
function CombineDofs(componentX::Int,componentY::Int,dofsX::Array{Int,1},dofsY::Array{Int,1})
    @assert length(dofsX) == length(dofsY)
    return CombineDofs("CombineDofs[$componentX,$componentY] (ndofs = $(length(dofsX)))",componentX,componentY,dofsX,dofsY, AssemblyAlways)
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
    A::FEMatrix,
    b::FEVector,
    Constraint::CombineDofs,
    Target::FEVector;
    current_equations = "all")

    fixed_dofs = []

    c = Constraint.componentX
    c2 = Constraint.componentY
    @logmsg DeepInfo "Combining dofs of component $c and $c2..."
    
    # add subblock [dofsY,dofsY] of block [c2,c2] to subblock [dofsX,dofsX] of block [c,c]
    # and penalize dofsY dofs
    rows = rowvals(A.entries.cscmatrix)
    targetrow = 0
    sourcerow = 0
    targetcolumn = 0
    sourcecolumn = 0
    for dof = 1 :length(Constraint.dofsX)

        targetrow = A[c,c].offsetX + Constraint.dofsX[dof]
        sourcerow = A[c2,c2].offsetX + Constraint.dofsY[dof]
        #println("copying sourcerow=$sourcerow to targetrow=$targetrow")
        for dof = 1 : length(Constraint.dofsX)
            sourcecolumn = Constraint.dofsY[dof] + A[c2,c2].offsetY
            for r in nzrange(A.entries.cscmatrix, sourcecolumn)
                if sourcerow == rows[r]
                    targetcolumn = Constraint.dofsX[dof] + A[c,c].offsetY
                    A.entries[targetrow, targetcolumn] += 0.5*A.entries.cscmatrix.nzval[r] 
                end
            end
        end
        targetcolumn = A[c,c].offsetY + Constraint.dofsX[dof]
        sourcecolumn = A[c2,c2].offsetY + Constraint.dofsY[dof]
        #println("copying sourcecolumn=$sourcecolumn to targetcolumn=$targetcolumn")
        for dof = 1 : length(Constraint.dofsX)
            sourcerow = Constraint.dofsY[dof] + A[c2,c2].offsetX
            for r in nzrange(A.entries.cscmatrix, sourcecolumn)
                if sourcerow == rows[r]
                    targetrow = Constraint.dofsX[dof] + A[c,c].offsetX
                    A.entries[targetrow,targetcolumn] += 0.5*A.entries.cscmatrix.nzval[r] 
                end
            end
        end

        # fix one of the dofs
        Target.entries[sourcecolumn] = 0
        push!(fixed_dofs,sourcecolumn)
    end
    return fixed_dofs
end

function realize_constraint!(
    Target::FEVector,
    Constraint::FixedIntegralMean)

    c = Constraint.component
    
    @logmsg MoreInfo "Moving integral mean for component $c to value $(Constraint.value)"

    # move integral mean
    pmeanIntegrator = ItemIntegrator(Float64,ON_CELLS,[Identity])
    meanvalue =  evaluate(pmeanIntegrator,Target[c])
    total_area = sum(Target.FEVectorBlocks[c].FES.xgrid[CellVolumes], dims=1)[1]
    meanvalue /= total_area
    for dof=1:Target.FEVectorBlocks[c].FES.ndofs
        Target[c][dof] -= meanvalue + Constraint.value
    end    
end


function realize_constraint!(
    Target::FEVector,
    Constraint::CombineDofs)

    c = Constraint.componentX
    c2 = Constraint.componentY
    @debug "Moving entries of combined dofs from component $c to component $c2"

    for dof = 1 : length(Constraint.dofsX)
        Target[c2][Constraint.dofsY[dof]] = Target[c][Constraint.dofsX[dof]]
    end 
end


