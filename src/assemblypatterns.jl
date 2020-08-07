###########################
# AbstractAssemblyPattern #
###########################

# provides several patterns for finite element assembly

# ItemIntegrator
#   - functional that computes piecewise integrals
#     for each item: I(item) = Int_item action(op(*)) dx
#   - can be called with evaluate! to compute piecewise integrals (of a given FEBlock)
#   - can be called with evaluate to compute total integral (of a given FEBlock)
#   - constructor is independent of FE or grid!



"""
$(TYPEDEF)

abstract type for assembly patterns
"""
abstract type AbstractAssemblyPattern{T <: Real, AT<:AbstractAssemblyType} end

"""
$(TYPEDEF)

assembly pattern linear form (that only depends on one quantity)
"""
struct LinearForm{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    FE::FESpace
    operator::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    regions::Array{Int,1}
end   

"""
````
function LinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::FESpace,
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    regions::Array{Int,1} = [0])
````

Creates a LinearForm.
"""
function LinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::FESpace,
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    regions::Array{Int,1} = [0])
    return LinearForm{T,AT}(FE,operator,action,regions)
end

"""
$(TYPEDEF)

assembly pattern bilinear form (that depends on two quantities)
"""
struct BilinearForm{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    FE1::FESpace
    FE2::FESpace
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction # is only applied to FE1/operator1
    regions::Array{Int,1}
    symmetric::Bool
end   

"""
````
function BilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    FE2::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
````

Creates an unsymmetric BilinearForm.
"""
function BilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    FE2::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
    return BilinearForm{T,AT}(FE1,FE2,operator1,operator2,action,regions,false)
end

"""
````
function SymmetricBilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
````

Creates a symmetric BilinearForm that can be assembled into a matrix or a vector (with one argument fixed)
"""
function SymmetricBilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
    return BilinearForm{T,AT}(FE1,FE1,operator1,operator1,action,regions,true)
end


"""
$(TYPEDEF)

assembly pattern trilinear form (that depends on three quantities)
"""
struct TrilinearForm{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    FE1::FESpace # < --- FE position that has to be fixed by an FEVectorBlock during assembly
    FE2::FESpace
    FE3::FESpace
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    operator3::Type{<:AbstractFunctionOperator}
    action::AbstractAction # is only applied to FE2/operator2
    regions::Array{Int,1}
end   

"""
````
function TrilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    FE2::FESpace,
    FE3::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    operator3::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1 + FE2/operator2
    regions::Array{Int,1} = [0])
````

Creates a TrilinearForm that can be assembeld into a matrix (with one argument fixed) or into a vector (with two fixed arguments).
"""
function TrilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::FESpace,
    FE2::FESpace,
    FE3::FESpace,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    operator3::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1 + FE2/operator2
    regions::Array{Int,1} = [0])
    return TrilinearForm{T,AT}(FE1,FE2,FE3,operator1,operator2,operator3,action,regions)
end

"""
$(TYPEDEF)

assembly pattern item integrator that can e.g. be used for error/norm evaluations
"""
struct ItemIntegrator{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    operator::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    regions::Array{Int,1}
end



# junctions for dof fields
FEPropertyDofs4AssemblyType(FE::FESpace,::Type{AssemblyTypeCELL}) = FE.CellDofs
FEPropertyDofs4AssemblyType(FE::FESpace,::Type{AssemblyTypeFACE}) = FE.FaceDofs
FEPropertyDofs4AssemblyType(FE::FESpace,::Type{AssemblyTypeBFACE}) = FE.BFaceDofs
FEPropertyDofs4AssemblyType(FE::FESpace,::Type{AssemblyTypeBFACECELL}) = FE.CellDofs


# unique functions that only selects uniques in specified regions
function Base.unique(xItemGeometries, xItemRegions, xItemDofs, regions)
    nitems = 0
    try
        nitems = num_sources(xItemGeometries)
    catch
        nitems = length(xItemGeometries)
    end      
    EG::Array{DataType,1} = []
    ndofs4EG = Array{Array{Int,1},1}(undef,length(xItemDofs))
    for e = 1 : length(xItemDofs)
        ndofs4EG[e] = []
    end
    iEG = 0
    cellEG = Triangle2D
    for item = 1 : nitems
        for j = 1 : length(regions)
            if xItemRegions[item] == regions[j]
                cellEG = xItemGeometries[item]
                iEG = 0
                for k = 1 : length(EG)
                    if cellEG == EG[k]
                        iEG = k
                        break;
                    end
                end
                if iEG == 0
                    append!(EG, [xItemGeometries[item]])
                    for e = 1 : length(xItemDofs)
                        append!(ndofs4EG[e], num_targets(xItemDofs[e],item))
                    end
                end  
                break; # rest of for loop can be skipped
            end    
        end
    end    
    return EG, ndofs4EG
end


function prepareOperatorAssembly(
    form::AbstractAssemblyPattern{T,AT},
    operator::Array{DataType,1},
    FE::Array{<:FESpace,1},
    regions::Array{Int32,1},
    nrfactors::Int,
    bonus_quadorder::Int,
    verbosity::Int) where {T<: Real, AT <: AbstractAssemblyType}

    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemRegions = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency,SerialVariableTargetAdjacency},1}(undef,length(FE))
    for j=1:length(FE)
        xItemDofs[j] = FEPropertyDofs4AssemblyType(FE[j],AT)
    end    

    # find unique ElementGeometries
    EG, ndofs4EG = Base.unique(xItemGeometries, xItemRegions, xItemDofs, regions)

    # find proper quadrature QuadratureRules
    # and construct matching FEBasisEvaluators
    qf = Array{QuadratureRule,1}(undef,length(EG))
    basisevaler = Array{Array{FEBasisEvaluator,1},1}(undef,length(EG))
    quadorder = 0
    for j = 1 : length(EG)
        basisevaler[j] = Array{FEBasisEvaluator,1}(undef,length(FE))
        quadorder = 0
        # choose quadrature order for all finite elements
        for k = 1 : length(FE)
            FEType = eltype(FE[k])
            quadorder = max(quadorder,bonus_quadorder + nrfactors*(get_polynomialorder(FEType, EG[j]) + QuadratureOrderShift4Operator(FEType,operator[k])))
        end
        for k = 1 : length(FE)
            if k > 1 && FE[k] == FE[1] && operator[k] == operator[1]
                basisevaler[j][k] = basisevaler[j][1] # e.g. for symmetric bilinerforms
            else    
                qf[j] = QuadratureRule{T,EG[j]}(quadorder);
                basisevaler[j][k] = FEBasisEvaluator{T,eltype(FE[k]),EG[j],operator[k],AT}(FE[k], qf[j]; verbosity = verbosity)
            end    
        end    
    end        
    if verbosity > 1
        println("\nASSEMBLY PREPARATION $(typeof(form))")
        println("====================================")
        for k = 1 : length(FE)
            println("      FE[$k] = $(FE[k].name), ndofs = $(FE[k].ndofs)")
            println("operator[$k] = $(operator[k])")
        end    
        println("     action = $(form.action)")
        println("    regions = $regions")
        println("   uniqueEG = $EG")
        for j = 1 : length(EG)
            println("\nQuadratureRule [$j] for $(EG[j]):")
            Base.show(qf[j])
        end
    end
    dofitem4item(item) = item
    EG4item(item) = xItemGeometries[item]
    function FEevaler4item(target,item) 
        return nothing # we assume that target is already 1:length(FE) which stays the same for all items
    end
    return EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, FEevaler4item
end

# dysfunctional at the moment
# will be repaired when assembly design steps are decided
# function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AssemblyTypeBFACECELL}, operator::Type{<:AbstractFunctionOperator}, FE::AbstractFiniteElement, regions::Array{Int32,1}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, verbosity::Int)
#     # find proper quadrature QuadratureRules
#     xItemGeometries = FE.xgrid[GridComponentGeometries4AssemblyType(AT)]
#     xCellGeometries = FE.xgrid[CellGeometries]

#     # find unique ElementGeometries
#     # todo: we need unique CellGeometriy/BFaceGeometrie pairs in BFace regions

#     qf = Array{QuadratureRule,1}(undef,length(EG))
#     basisevaler = Array{Array{FEBasisEvaluator,1},1}(undef,length(EG))
#     quadorder = 0
#     nfaces4cell = 0
#     for j = 1 : length(EG)
#         itemEG = facetype_of_cellface(EG[j],1)
#         nfaces4cell = nfaces_for_geometry(EG[j])
#         basisevaler[j] = Array{FEBasisEvaluator,1}(undef,nfaces4cell)
#         quadorder = bonus_quadorder + nrfactors*get_polynomialorder(typeof(FE), itemEG)
#         qf[j] = QuadratureRule{T,itemEG}(quadorder);
#         qfxref = qf[j].xref
#         xrefFACE2CELL = xrefFACE2xrefCELL(EG[j])
#         for k = 1 : nfaces4cell
#             for i = 1 : length(qfxref)
#                 qf[j].xref[i] = xrefFACE2CELL[k](qfxref[i])
#             end
#             basisevaler[j][k] = FEBasisEvaluator{T,typeof(FE),EG[j],operator,AT}(FE, qf[j])
#         end    
#         for i = 1 : length(qfxref)
#             qf[j].xref[i] = qfxref[i]
#         end
#     end        
#     if verbosity > 0
#         println("\nASSEMBLY PREPARATION $form")
#         println("====================================")
#         println("        FE = $(FE.name), ndofs = $(FE.ndofs)")
#         println("  operator = $operator")
#         println("   regions = $regions")
#         println("  uniqueEG = $EG")
#         for j = 1 : length(EG)
#             println("\nQuadratureRule [$j] for $(EG[j]):")
#             show(qf[j])
#         end
#     end
#     dofitem4item(item) = FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]
#     EG4item(item) = xCellGeometries[FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]]
#     FEevaler4item(item) = FE.xgrid[BFaceCellPos][item]
#     return EG, qf, basisevaler, EG4item , dofitem4item, FEevaler4item
# end



"""
````
function evaluate!(
    b::AbstractArray{<:Real,2},
    form::ItemIntegrator{T,AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}
````

Evaluation of an ItemIntegrator form with given FEVectorBlock FEB into given two-dimensional Array b.
"""
function evaluate!(
    b::AbstractArray{<:Real,2},
    form::ItemIntegrator{T,AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

    FE = FEB.FES
    FEType = eltype(FE)
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)

    operator = form.operator
    regions = form.regions
    action = form.action
    bonus_quadorder = action.bonus_quadorder
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(form, [operator], [FE], regions, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][1].cvals,1)
    @assert size(b,2) == cvals_resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = [1] # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    coeffs = zeros(Float64,max_num_targets_per_source(xItemDofs))
    action_input = zeros(T,cvals_resultdim) # heap for action input
    action_result = zeros(T,action.resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        for j=1:ndofs4item
            coeffs[j] = FEB[xItemDofs[j,dofitem]]
        end

        weights = qf[iEG].w
        for i in eachindex(weights)
            # apply action to FEVector
            fill!(action_input,0)
            eval!(action_input,basisevaler[iEG][evalnr[1]],coeffs, i)
            apply_action!(action_result, action_input, action, i)
            for j = 1 : action.resultdim
                b[item,j] += action_result[j] * weights[i] * xItemVolumes[item]
            end
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end

"""
````
function evaluate(
    form::ItemIntegrator{T,AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

````

Evaluation of an ItemIntegrator form with given FEVectorBlock FEB, only returns accumulation over all items.
"""
function evaluate(
    form::ItemIntegrator{T,AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

    FE = FEB.FES
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)

    operator = form.operator
    regions = form.regions
    action = form.action
    bonus_quadorder = action.bonus_quadorder
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(form, [operator], [FE], regions, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][1].cvals,1)

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item::Int = 0 # number of dofs for item
    evalnr = [1] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    coeffs = zeros(T,max_num_targets_per_source(xItemDofs))
    action_input = zeros(T,cvals_resultdim) # heap for action input
    action_result = zeros(T,action.resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    result = 0.0
    nregions::Int = length(regions)
    dof = 0
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        for j=1:ndofs4item
            coeffs[j] = FEB[xItemDofs[j,dofitem]]
        end

        weights = qf[iEG].w
        for i = 1 : length(weights)
            # apply action to FEVector
            fill!(action_input,0)
            eval!(action_input,basisevaler[iEG][evalnr[1]],coeffs, i)
            apply_action!(action_result, action_input, action, i)
            for j = 1 : action.resultdim
                result += action_result[j] * weights[i] * xItemVolumes[item]
            end
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return result
end


"""
````
assemble!(
    b::AbstractArray{<:Real,2},
    LF::LinearForm{T,AT};
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

````

Assembly of a LinearForm LF into given two-dimensional Array b.
"""
function assemble!(
    b::AbstractArray{<:Real,2},
    LF::LinearForm{T,AT};
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}
    FE = LF.FE
    operator = LF.operator
    action = LF.action
    regions = LF.regions

    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    bonus_quadorder = action.bonus_quadorder
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(LF, [operator], [FE], regions, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][1].cvals,1)
    action_resultdim::Int = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item::Int = 0 # number of dofs for item
    evalnr = [1] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    maxdofs::Int = max_num_targets_per_source(xItemDofs)
    dofs = zeros(Int32,maxdofs)
    temp::T = 0 # some temporary variable
    action_input = zeros(T,cvals_resultdim) # heap for action input
    action_result = zeros(T,action_resultdim) # heap for action output
    basisvals::Array{T,3} = basisevaler[1][1].cvals # pointer to operator results
    localb = zeros(T,maxdofs,action_resultdim)
    nregions::Int = length(regions)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        basisvals = basisevaler[iEG][evalnr[1]].cvals

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        for j=1:ndofs4item
            dofs[j] = xItemDofs[j,dofitem]
        end

        weights = qf[iEG].w
        for i in eachindex(weights)
            for dof_i = 1 : ndofs4item
                # apply action
                eval!(action_input,basisevaler[iEG][evalnr[1]], dof_i, i)
                apply_action!(action_result, action_input, action, i)
                for j = 1 : action_resultdim
                   localb[dof_i,j] += action_result[j] * weights[i]
                end
            end 
        end  

        for dof_i = 1 : ndofs4item, j = 1 : action_resultdim
            b[dofs[dof_i],j] += localb[dof_i,j] * xItemVolumes[item]
        end
        fill!(localb, 0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end


"""
````
assemble!(
    b::AbstractArray{T,1},
    LF::LinearForm{T,AT}; # LF has to be scalar-valued
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

````

Assembly of a LinearForm LF into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
"""
function assemble!( # LF has to have resultdim == 1
    b::AbstractArray{T,1},
    LF::LinearForm{T,AT};
    verbosity::Int = 0,
    factor::Real = 1) where {T<: Real, AT <: AbstractAssemblyType}
    FE = LF.FE
    operator = LF.operator
    action = LF.action
    regions = LF.regions

    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE.xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs = FEPropertyDofs4AssemblyType(FE,AT)
    xItemRegions = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    bonus_quadorder = action.bonus_quadorder
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(LF, [operator], [FE], regions, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE)
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][1].cvals,1)
    action_resultdim::Int = action.resultdim
    @assert action_resultdim == 1

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item::Int = 0 # number of dofs for item
    evalnr = [1] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    maxdofs::Int = max_num_targets_per_source(xItemDofs)
    dofs = zeros(Int,maxdofs)
    temp::T = 0 # some temporary variable
    action_input = zeros(T,cvals_resultdim) # heap for action input
    action_result = zeros(T,action_resultdim) # heap for action output
    basisvals::Array{T,3} = basisevaler[1][1].cvals # pointer to operator results
    localb = zeros(T,maxdofs)
    nregions::Int = length(regions)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        basisvals = basisevaler[iEG][evalnr[1]].cvals

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        for j=1:ndofs4item
            dofs[j] = xItemDofs[j,dofitem]
        end
        
        weights = qf[iEG].w
        for i in eachindex(weights)
            for dof_i = 1 : ndofs4item
                eval!(action_input,basisevaler[iEG][evalnr[1]], dof_i, i)
                apply_action!(action_result, action_input, action, i)
                localb[dof_i] += action_result[1] * weights[i]
            end 
        end  

        for dof_i = 1 : ndofs4item
            b[dofs[dof_i]] += factor * localb[dof_i] * xItemVolumes[item]
        end
        fill!(localb, 0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end


"""
````
assemble!(
    A::AbstractArray{<:Real,2},
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor::Real = 1,
    transposed_assembly::Bool = false,
    transpose_copy = Nothing) where {T<: Real, AT <: AbstractAssemblyType}
````

Assembly of a BilinearForm BLF into given two-dimensional AbstractArray (e.g. FEMatrixBlock).
"""
function assemble!(
    A::AbstractArray{<:Real,2},
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor::Real = 1,
    transposed_assembly::Bool = false,
    transpose_copy = Nothing) where {T<: Real, AT <: AbstractAssemblyType}

    FE = [BLF.FE1, BLF.FE2]
    operator = [BLF.operator1, BLF.operator2]
    action = BLF.action
    bonus_quadorder = BLF.action.bonus_quadorder
    regions = BLF.regions

    
    # collect grid information
    xItemNodes = FE[1].xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs1 = FEPropertyDofs4AssemblyType(FE[1],AT)
    xItemDofs2 = FEPropertyDofs4AssemblyType(FE[2],AT)
    xItemRegions = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = Int64(num_sources(xItemNodes))
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(BLF, operator, FE, regions, 2, bonus_quadorder, verbosity-1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE[1])
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][apply_action_to].cvals,1)
    action_resultdim::Int = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    evalnr = [1,2] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    dofs = zeros(Int,maxdofs1)
    dofs2 = zeros(Int,maxdofs2)
    temp::T = 0 # some temporary variable
    action_input = zeros(T,cvals_resultdim) # heap for action input
    action_result = zeros(T,action_resultdim) # heap for action output
    localmatrix = zeros(T,maxdofs1,maxdofs2)
    basisvals::Array{T,3} = basisevaler[1][1].cvals
    basisvals2::Array{T,3} = basisevaler[1][2].cvals
    nregions::Int = length(regions)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item1 = ndofs4EG[1][iEG]
        ndofs4item2 = ndofs4EG[2][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        update!(basisevaler[iEG][evalnr[2]],dofitem)
        basisvals = basisevaler[iEG][evalnr[1]].cvals
        basisvals2 = basisevaler[iEG][evalnr[2]].cvals

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        for j=1:ndofs4item1
            dofs[j] = xItemDofs1[j,dofitem]
        end
        for j=1:ndofs4item2
            dofs2[j] = xItemDofs2[j,dofitem]
        end

        weights = qf[iEG].w
        for i in eachindex(weights)
           
            if apply_action_to == 1
                for dof_i = 1 : ndofs4item1
                    eval!(action_input,basisevaler[iEG][evalnr[1]], dof_i, i)
                    apply_action!(action_result, action_input, action, i)

                    if BLF.symmetric == false
                        for dof_j = 1 : ndofs4item2
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k]*basisvals2[k,dof_j,i]
                            end
                            localmatrix[dof_i,dof_j] += temp * weights[i]
                        end
                    else # symmetric case
                        for dof_j = dof_i : ndofs4item2
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k]*basisvals2[k,dof_j,i]
                            end
                            localmatrix[dof_i,dof_j] += temp * weights[i]
                        end
                    end
                end 
            else
                for dof_j = 1 : ndofs4item2
                    eval!(action_input,basisevaler[iEG][evalnr[2]], dof_j, i)
                    apply_action!(action_result, action_input, action, i)

                    if BLF.symmetric == false
                        for dof_i = 1 : ndofs4item1
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k]*basisvals[k,dof_j,i]
                            end
                            localmatrix[dof_i,dof_j] += temp * weights[i]
                        end
                    else # symmetric case
                        for dof_i = dof_j : ndofs4item1
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k]*basisvals[k,dof_j,i]
                            end
                            localmatrix[dof_i,dof_j] += temp * weights[i]
                        end
                    end
                end 

            end
        end 

        # copy localmatrix into global matrix
        if BLF.symmetric == false
            for dof_i = 1 : ndofs4item1, dof_j = 1 : ndofs4item2
                if localmatrix[dof_i,dof_j] != 0
                    if transposed_assembly == true
                        A[dofs2[dof_j],dofs[dof_i]] += localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor   
                    else
                        A[dofs[dof_i],dofs2[dof_j]] += localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor     
                    end
                    if transpose_copy != Nothing # sign is changed in case nonzero rhs data is applied to LagrangeMultiplier (good idea?)
                        if transposed_assembly == true
                            transpose_copy[dofs[dof_i],dofs2[dof_j]] -= localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor
                        else
                            transpose_copy[dofs2[dof_j],dofs[dof_i]] -= localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor
                        end
                    end
                end
            end
        else # symmetric case
            for dof_i = 1 : ndofs4item1, dof_j = dof_i+1 : ndofs4item2
                if localmatrix[dof_i,dof_j] != 0 
                    temp = localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor
                    A[dofs2[dof_j],dofs[dof_i]] += temp
                    A[dofs2[dof_i],dofs[dof_j]] += temp
                end
            end    
            for dof_i = 1 : ndofs4item1
                A[dofs2[dof_i],dofs[dof_i]] += localmatrix[dof_i,dof_i] * xItemVolumes[item] * factor
            end    
        end    
        fill!(localmatrix,0.0)

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

end



"""
````
assemble!(
    b::AbstractArray{<:Real,1},
    fixedFE::FEVectorBlock,    # coefficient for fixed 2nd component
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    factor::Real = 1,
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}
````

Assembly of a BilinearForm BLF into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the second argument is fixed by the given coefficients in fixedFE.
With apply_action_to=2 the action can be also applied to the second fixed argument instead of the first one (default).
"""
function assemble!(
    b::AbstractArray{<:Real,1},
    fixedFE::FEVectorBlock,    # coefficient for fixed 2nd component
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    factor::Real = 1,
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

    FE = [BLF.FE1, BLF.FE2]
    operator = [BLF.operator1, BLF.operator2]
    action = BLF.action
    bonus_quadorder = BLF.action.bonus_quadorder
    regions = BLF.regions

    
    # collect grid information
    xItemNodes = FE[1].xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs1 = FEPropertyDofs4AssemblyType(FE[1],AT)
    xItemDofs2 = FEPropertyDofs4AssemblyType(FE[2],AT)
    xItemRegions = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = Int64(num_sources(xItemNodes))
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(BLF, operator, FE, regions, 2, bonus_quadorder, verbosity-1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE[1])
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][apply_action_to].cvals,1)
    action_resultdim::Int = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    evalnr = [1,2] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    dofs = zeros(Int,maxdofs1)
    coeffs2 = zeros(T,maxdofs2)
    temp::T = 0 # some temporary variable
    fixedval = zeros(T,action_resultdim)
    action_input = zeros(T,cvals_resultdim) # heap for action input
    action_result = zeros(T,action_resultdim) # heap for action output
    localb = zeros(T,maxdofs1)
    basisvals::Array{T,3} = basisevaler[1][1].cvals
    nregions::Int = length(regions)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item1 = ndofs4EG[1][iEG]
        ndofs4item2 = ndofs4EG[2][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        update!(basisevaler[iEG][evalnr[2]],dofitem)
        basisvals = basisevaler[iEG][evalnr[1]].cvals

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        for j=1:ndofs4item1
            dofs[j] = xItemDofs1[j,dofitem]
        end
        for j=1:ndofs4item2
            coeffs2[j] = fixedFE[xItemDofs2[j,dofitem]]
        end

        weights = qf[iEG].w
        for i in eachindex(weights)

            # evaluate second component
            fill!(fixedval,0.0)
            eval!(fixedval,basisevaler[iEG][evalnr[2]],coeffs2, i)

            if apply_action_to == 2
                # apply action to second argument
                apply_action!(action_result, fixedval, action, i)

                # multiply first argument
                for dof_i = 1 : ndofs4item1
                    temp = 0
                    for k = 1 : action_resultdim
                        temp += action_result[k]*basisvals[k,dof_i,i]
                    end
                    localb[dof_i] += temp * weights[i]
                end 
            else
                for dof_i = 1 : ndofs4item1
                    # apply action to first argument
                    eval!(action_input,basisevaler[iEG][evalnr[1]],dof_i, i)
                    apply_action!(action_result, action_input, action, i)
    
                    # multiply second argument
                    temp = 0
                    for k = 1 : action_resultdim
                        temp += action_result[k]*fixedval[k]
                    end
                    localb[dof_i] += temp * weights[i]
                end 
            end
        end 

        for dof_i = 1 : ndofs4item1
            b[dofs[dof_i]] += localb[dof_i] * xItemVolumes[item] * factor
        end
        fill!(localb, 0.0)

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end

"""
````
assemble!(
    assemble!(
    A::AbstractArray{<:Real,2},
    FE1::FEVectorBlock,
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    transposed_assembly::Bool = false,
    factor::Real = 1)
````

Assembly of a TrilinearForm TLF into given two-dimensional AbstractArray (e.g. a FEMatrixBlock).
Here, the first argument is fixed by the given coefficients in FE1.
"""
function assemble!(
    A::AbstractArray{<:Real,2},
    FE1::FEVectorBlock,
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    transposed_assembly::Bool = false,
    factor::Real = 1) where {T<: Real, AT <: AbstractAssemblyType}

    FE = [TLF.FE1, TLF.FE2, TLF.FE3]
    operator = [TLF.operator1, TLF.operator2, TLF.operator3]
    action = TLF.action
    bonus_quadorder = TLF.action.bonus_quadorder
    regions = TLF.regions

    
    # collect grid information
    xItemNodes = FE[1].xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs1 = FEPropertyDofs4AssemblyType(FE[1],AT)
    xItemDofs2 = FEPropertyDofs4AssemblyType(FE[2],AT)
    xItemDofs3 = FEPropertyDofs4AssemblyType(FE[3],AT)
    xItemRegions = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = Int64(num_sources(xItemNodes))
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(TLF, operator, FE, regions, 3, bonus_quadorder, verbosity-1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE[1])
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][1].cvals,1)
    cvals_resultdim2::Int = size(basisevaler[1][2].cvals,1)
    action_resultdim::Int = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    ndofs4item3::Int = 0 # number of dofs for item
    evalnr = [1,2,3] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    maxdofs3::Int = max_num_targets_per_source(xItemDofs3)
    coeffs = zeros(T,maxdofs1)
    dofs2 = zeros(Int,maxdofs2)
    dofs3 = zeros(Int,maxdofs3)
    temp::T = 0 # some temporary variable
    evalFE1 = zeros(T,cvals_resultdim) # heap for action input
    action_input = zeros(T,cvals_resultdim+cvals_resultdim2) # heap for action input
    action_result = zeros(T,action_resultdim) # heap for action output
    localmatrix = zeros(T,maxdofs2,maxdofs3)
    basisvals3::Array{T,3} = basisevaler[1][3].cvals
    nregions::Int = length(regions)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item1 = ndofs4EG[1][iEG]
        ndofs4item2 = ndofs4EG[2][iEG]
        ndofs4item3 = ndofs4EG[3][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        update!(basisevaler[iEG][evalnr[2]],dofitem)
        update!(basisevaler[iEG][evalnr[3]],dofitem)
        basisvals3 = basisevaler[iEG][evalnr[3]].cvals

        # update action
        update!(action, basisevaler[iEG][evalnr[2]], item, regions[r])

        # update dofs
        for j=1:ndofs4item1
            coeffs[j] = FE1[xItemDofs1[j,dofitem]]
        end
        for j=1:ndofs4item2
            dofs2[j] = xItemDofs2[j,dofitem]
        end
        for j=1:ndofs4item3
            dofs3[j] = xItemDofs3[j,dofitem]
        end

        weights = qf[iEG].w
        for i in eachindex(weights)

            # evaluate first component
            fill!(action_input,0.0)
            eval!(action_input,basisevaler[iEG][evalnr[1]],coeffs, i)
           
            for dof_i = 1 : ndofs4item2
                # apply action to FE1 eval and second argument
                eval!(action_input,basisevaler[iEG][evalnr[2]],dof_i, i, offset = cvals_resultdim)
                apply_action!(action_result, action_input, action, i)

                for dof_j = 1 : ndofs4item3
                    temp = 0
                    for k = 1 : action_resultdim
                         temp += action_result[k]*basisvals3[k,dof_j,i]
                    end
                    localmatrix[dof_i,dof_j] += temp * weights[i]
                end
            end 
        end 

        # copy localmatrix into global matrix
        for dof_i = 1 : ndofs4item2, dof_j = 1 : ndofs4item3
             if localmatrix[dof_i,dof_j] != 0
                if transposed_assembly == true
                    A[dofs3[dof_j],dofs2[dof_i]] += localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor
                else
                    A[dofs2[dof_i],dofs3[dof_j]] += localmatrix[dof_i,dof_j] * xItemVolumes[item] * factor 
                end
            end
        end
        fill!(localmatrix,0.0)

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end



"""
````
assemble!(
    assemble!(
    A::AbstractArray{<:Real,1},
    FE1::FEVectorBlock,
    FE2::FEVectorBlock.
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    factor::Real = 1)
````

Assembly of a TrilinearForm TLF into given two-dimensional AbstractArray (e.g. a FEMatrixBlock).
Here, the first two arguments are fixed by the given coefficients in FE1 and FE2.
"""
function assemble!(
    b::AbstractArray{<:Real,1},
    FE1::FEVectorBlock,
    FE2::FEVectorBlock,
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    factor::Real = 1) where {T<: Real, AT <: AbstractAssemblyType}

    FE = [TLF.FE1, TLF.FE2, TLF.FE3]
    operator = [TLF.operator1, TLF.operator2, TLF.operator3]
    action = TLF.action
    bonus_quadorder = TLF.action.bonus_quadorder
    regions = TLF.regions

    
    # collect grid information
    xItemNodes = FE[1].xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{Float64,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemDofs1 = FEPropertyDofs4AssemblyType(FE[1],AT)
    xItemDofs2 = FEPropertyDofs4AssemblyType(FE[2],AT)
    xItemDofs3 = FEPropertyDofs4AssemblyType(FE[3],AT)
    xItemRegions = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = Int64(num_sources(xItemNodes))
    
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item! = prepareOperatorAssembly(TLF, operator, FE, regions, 3, bonus_quadorder, verbosity-1)

    # collect FE and FEBasisEvaluator information
    FEType = eltype(FE[1])
    ncomponents::Int = get_ncomponents(FEType)
    cvals_resultdim::Int = size(basisevaler[1][1].cvals,1)
    cvals_resultdim2::Int = size(basisevaler[1][2].cvals,1)
    action_resultdim::Int = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG::Int = 1 # index to the correct unique geometry
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    ndofs4item3::Int = 0 # number of dofs for item
    evalnr = [1,2,3] # evaler number that has to be used for current item
    dofitem::Int = 0 # itemnr where the dof numbers can be found
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    maxdofs3::Int = max_num_targets_per_source(xItemDofs3)
    coeffs1 = zeros(T,maxdofs1)
    coeffs2 = zeros(T,maxdofs2)
    dofs3 = zeros(Int,maxdofs3)
    temp::T = 0 # some temporary variable
    evalFE1 = zeros(T,cvals_resultdim) # heap for action input
    action_input = zeros(T,cvals_resultdim+cvals_resultdim2) # heap for action input
    action_result = zeros(T,action_resultdim) # heap for action output
    localb = zeros(T,maxdofs3)
    basisvals3::Array{T,3} = basisevaler[1][3].cvals
    nregions::Int = length(regions)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations

    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item1 = ndofs4EG[1][iEG]
        ndofs4item2 = ndofs4EG[2][iEG]
        ndofs4item3 = ndofs4EG[3][iEG]

        # update FEbasisevaler
        evaler4item!(evalnr,item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        update!(basisevaler[iEG][evalnr[2]],dofitem)
        update!(basisevaler[iEG][evalnr[3]],dofitem)
        basisvals3 = basisevaler[iEG][evalnr[3]].cvals

        # update action
        update!(action, basisevaler[iEG][evalnr[2]], item, regions[r])

        # update dofs
        for j=1:ndofs4item1
            coeffs[j] = FE1[xItemDofs1[j,dofitem]]
        end
        for j=1:ndofs4item2
            coeff2[j] = FE2[xItemDofs2[j,dofitem]]
        end
        for j=1:ndofs4item3
            dofs3[j] = xItemDofs3[j,dofitem]
        end

        weights = qf[iEG].w
        for i in eachindex(weights)

            # evaluate first and second component
            fill!(action_input,0.0)
            eval!(action_input,basisevaler[iEG][evalnr[1]],coeffs, i)
            eval!(action_input,basisevaler[iEG][evalnr[2]],coeffs2, i, offset = cvals_resultdim)

            # apply action to FE1 and FE2
            apply_action!(action_result, action_input, action, i)
           
            # multiply third component
            for dof_j = 1 : ndofs4item2
                temp = 0
                for k = 1 : action_resultdim
                    temp += action_result[k]*basisvals3[k,dof_j,i]
                end
                localb[dof_j] += temp * weights[i]
            end 
        end 

        for dof_i = 1 : ndofs4item1
            b[dofs[dof_i]] += localb[dof_i] * xItemVolumes[item] * factor
        end
        fill!(localb, 0.0)

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end



"""
````
function L2ErrorIntegrator(
    exact_function::Function,
    operator::Type{<:AbstractFunctionOperator},
    xdim::Int,
    ncomponents::Int = 1;
    AT::Type{<:AbstractAssemblyType} = AssemblyTypeCELL,
    bonus_quadorder::Int = 0)
````

Creates an ItemIntegrator that compares FEVectorBlock operator-evaluations against the given exact_function and returns the L2-error.
"""
function L2ErrorIntegrator(
    exact_function::Function,
    operator::Type{<:AbstractFunctionOperator},
    xdim::Int,
    ncomponents::Int = 1;
    AT::Type{<:AbstractAssemblyType} = AssemblyTypeCELL,
    bonus_quadorder::Int = 0,
    time_dependent_data::Bool = false,
    time = 0)
    function L2error_function()
        temp = zeros(Float64,ncomponents)
        function closure(result,input,x)
            if time_dependent_data
                exact_function(temp,x,time)
            else
                exact_function(temp,x)
            end
            result[1] = 0.0
            for j=1:length(temp)
                result[1] += (temp[j] - input[j])^2
            end    
        end
    end    
    L2error_action = XFunctionAction(L2error_function(),1,xdim; bonus_quadorder = bonus_quadorder)
    return ItemIntegrator{Float64,AT}(operator, L2error_action, [0])
end