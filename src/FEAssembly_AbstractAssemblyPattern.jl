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

# LinearForm
#   - functional that depend on one FE function, i.e.
#     for each v_h: F(v_h) = Int_regions action(operator(v_h)) dx
#   - action (like multiplying coefficients, user-defined operation)
#     may be applied after evaluation of the standard operator

# BilinearForm
#   - functional that depend on two FE function, i.e.
#     for each u_h,v_h: A(u_h,v_h) = Int_regions action(op1(u_h))*op2(v_h) dx
#   - constructor for symmetric bilinearforms where FEType of u_h and v_h are the same
#     is also available (and everything should be written such that double evaluations are avoided)
#
#   - maybe in future: fixing on component with given FE function to define a LinearForm
#                      automatically

# maybe in future: TrilinearForm
#   - functional that depend on three FE function, i.e.
#     for each a_h,u_h,v_h: C(a_h,u_h,v_h) = Int_regions action(op1(a_h))*op2(u_h)*op3(v_h) dx
#   - assembly! where a_h (or also u_h) is fixed with given FEBlock


abstract type AbstractAssemblyPattern{AT<:AbstractAssemblyType} end

struct LinearForm{AT} <: AbstractAssemblyPattern{AT}
    FE::AbstractFiniteElement
    operator::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    regions::Array{Int,1}
end   

function LinearForm(AT::Type{<:AbstractAssemblyType},
    FE::AbstractFiniteElement,
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    regions::Array{Int,1} = [0])
    return LinearForm{AT}(FE,operator,action,regions)
end

struct BilinearForm{AT} <: AbstractAssemblyPattern{AT}
    FE1::AbstractFiniteElement
    FE2::AbstractFiniteElement
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction # is only applied to FE1/operator1
    regions::Array{Int,1}
    symmetric::Bool
end   
function BilinearForm(AT::Type{<:AbstractAssemblyType},
    FE1::AbstractFiniteElement,
    FE2::AbstractFiniteElement,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
    return BilinearForm{AT}(FE1,FE2,operator1,operator2,action,regions,false)
end
function SymmetricBilinearForm(AT::Type{<:AbstractAssemblyType},
    FE1::AbstractFiniteElement,
    operator1::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
    return BilinearForm{AT}(FE1,FE1,operator1,operator1,action,regions,true)
end


struct ItemIntegrator{AT} <: AbstractAssemblyPattern{AT}
    operator::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    regions::Array{Int,1}
end
function ItemIntegrator(AT::Type{<:AbstractAssemblyType},
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction; # is only applied to FE1/operator1
    regions::Array{Int,1} = [0])
    return ItemIntegrator{AT}(operator,action,regions)
end   

export AbstractAssemblyPattern,ItemIntegrator,LinearForm,BilinearForm,SymmetricBilinearForm
export assemble!, evaluate!, evaluate
export L2ErrorIntegrator


# junctions for dof fields
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeCELL}) = FE.CellDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeFACE}) = FE.FaceDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeBFACE}) = FE.BFaceDofs
FEPropertyDofs4AssemblyType(FE::AbstractFiniteElement,::Type{AbstractAssemblyTypeBFACECELL}) = FE.CellDofs


# unique functions that only selects uniques in specified regions
function unique(xItemGeometries, xItemRegions, xItemDofs, regions)
    nitems = 0
    try
        nitems = num_sources(xItemGeometries)
    catch
        nitems = length(xItemGeometries)
    end      
    EG::Array{DataType,1} = []
    ndofs4EG = Array{Array{Int32,1},1}(undef,length(xItemDofs))
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
    form::AbstractAssemblyPattern{AT},
    operator::Array{DataType,1},
    FE::Array{<:AbstractFiniteElement,1},
    regions::Array{Int32,1},
    NumberType::Type{<:Real},
    nrfactors::Int,
    bonus_quadorder::Int,
    verbosity::Int) where AT <: AbstractAssemblyType

    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    xItemRegions = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    xItemDofs = Array{VariableTargetAdjacency,1}(undef,length(FE))
    for j=1:length(FE)
        xItemDofs[j] = FEPropertyDofs4AssemblyType(FE[j],AT)
    end    

    # find unique ElementGeometries
    EG, ndofs4EG = unique(xItemGeometries, xItemRegions, xItemDofs, regions)

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
            quadorder = max(quadorder,bonus_quadorder + nrfactors*(FiniteElements.get_polynomialorder(typeof(FE[k]), EG[j]) + QuadratureOrderShift4Operator(typeof(FE[k]),operator[k])))
        end
        for k = 1 : length(FE)
            if k > 1 && FE[k] == FE[1] && operator[k] == operator[1]
                basisevaler[j][k] = basisevaler[j][1] # e.g. for symmetric bilinerforms
            else    
                qf[j] = QuadratureRule{NumberType,EG[j]}(quadorder);
                basisevaler[j][k] = FEBasisEvaluator{NumberType,typeof(FE[k]),EG[j],operator[k],AT}(FE[k], qf[j])
            end    
        end    
    end        
    if verbosity > 0
        println("\nASSEMBLY PREPARATION $(typeof(form))")
        println("====================================")
        for k = 1 : length(FE)
            println("      FE[$k] = $(FE[k].name), ndofs = $(FE[k].ndofs)")
            println("operator[$k] = $(operator[k])")
        end    
        println("    regions = $regions")
        println("   uniqueEG = $EG")
        for j = 1 : length(EG)
            println("\nQuadratureRule [$j] for $(EG[j]):")
            QuadratureRules.show(qf[j])
        end
    end
    dofitem4item(item) = item
    EG4item(item) = xItemGeometries[item]
    FEevaler4item(item) = 1:length(FE)
    return EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, FEevaler4item
end

# dysfunctional at the moment
# will be repaired when assembly design steps are decided
# function prepareOperatorAssembly(form::Type{<:AbstractFEForm}, AT::Type{<:AbstractAssemblyTypeBFACECELL}, operator::Type{<:AbstractFunctionOperator}, FE::AbstractFiniteElement, regions::Array{Int32,1}, NumberType::Type{<:Real}, nrfactors::Int, bonus_quadorder::Int, verbosity::Int)
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
#         nfaces4cell = nfaces_per_cell(EG[j])
#         basisevaler[j] = Array{FEBasisEvaluator,1}(undef,nfaces4cell)
#         quadorder = bonus_quadorder + nrfactors*FiniteElements.get_polynomialorder(typeof(FE), itemEG)
#         qf[j] = QuadratureRule{NumberType,itemEG}(quadorder);
#         qfxref = qf[j].xref
#         xrefFACE2CELL = xrefFACE2xrefCELL(EG[j])
#         for k = 1 : nfaces4cell
#             for i = 1 : length(qfxref)
#                 qf[j].xref[i] = xrefFACE2CELL[k](qfxref[i])
#             end
#             basisevaler[j][k] = FEBasisEvaluator{NumberType,typeof(FE),EG[j],operator,AT}(FE, qf[j])
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
#             QuadratureRules.show(qf[j])
#         end
#     end
#     dofitem4item(item) = FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]
#     EG4item(item) = xCellGeometries[FE.xgrid[FaceCells][1,FE.xgrid[BFaces][item]]]
#     FEevaler4item(item) = FE.xgrid[BFaceCellPos][item]
#     return EG, qf, basisevaler, EG4item , dofitem4item, FEevaler4item
# end

function evaluate!(
    b::AbstractArray{<:Real,1},
    form::ItemIntegrator{AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where AT <: AbstractAssemblyType

    NumberType = eltype(b)
    FE = FEB.FEType
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
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, [operator], [FE], regions, NumberType, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)
    @assert size(b,2) == cvals_resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action.resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEAssemblyvalue at quadrature point i

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr], item, regions[r])

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        for i in eachindex(qf[iEG].w)
            cvali = basisevaler[iEG][evalnr].cvals[i]
            # apply action to FEVector
            fill!(action_input,0)
            for dof_i = 1 : ndofs4item
                for k = 1 : cvals_resultdim
                    action_input[k] += FEB[dofs[dof_i]] * cvali[k,dof_i]
                end    
            end 
            apply_action!(action_result, action_input, action, i)
            for j = 1 : action.resultdim
                b[item,j] += action_result[j] * qf[iEG].w[i] * xItemVolumes[item]
            end
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end

function evaluate(
    form::ItemIntegrator{AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where AT <: AbstractAssemblyType

    NumberType = Float64
    FE = FEB.FEType
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
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(form, [operator], [FE], regions, NumberType, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = 0 # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action.resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEAssemblyvalue at quadrature point i

    result = 0.0
    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        for i in eachindex(qf[iEG].w)
            cvali = basisevaler[iEG][evalnr[1]].cvals[i]
            # apply action to FEVector
            fill!(action_input,0)
            for dof_i = 1 : ndofs4item
                for k = 1 : cvals_resultdim
                    action_input[k] += FEB[dofs[dof_i]] * cvali[k,dof_i]
                end    
            end 
            apply_action!(action_result, action_input, action, i)
            for j = 1 : action.resultdim
                result += action_result[j] * qf[iEG].w[i] * xItemVolumes[item]
            end
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return result
end

function assemble!(
    b::AbstractArray{<:Real,2},
    LF::LinearForm{AT};
    verbosity::Int = 0) where AT <: AbstractAssemblyType
    FE = LF.FE
    operator = LF.operator
    action = LF.action
    regions = LF.regions

    NumberType = eltype(b)
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
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(LF, [operator], [FE], regions, NumberType, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)
    action_resultdim = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = [0] # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    maxdofs = max_num_targets_per_source(xItemDofs)
    dofs = zeros(Int32,maxdofs)
    temp = 0 # some temporary variable
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action_resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEAssemblyvalue at quadrature point i
    localb = zeros(NumberType,maxdofs,action_resultdim)
    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        for i in eachindex(qf[iEG].w)
            cvali = basisevaler[iEG][evalnr[1]].cvals[i]

            for dof_i = 1 : ndofs4item
                # apply action
                for k = 1 : cvals_resultdim
                    action_input[k] = cvali[k,dof_i]
                end    
                apply_action!(action_result, action_input, action, i)

                for j = 1 : action_resultdim
                   localb[dof_i,j] += action_result[j] * qf[iEG].w[i]
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


function assemble!( # LF has to have resultdim == 1
    b::FEVectorBlock,
    LF::LinearForm{AT};
    verbosity::Int = 0) where AT <: AbstractAssemblyType
    FE = LF.FE
    operator = LF.operator
    action = LF.action
    regions = LF.regions

    NumberType = eltype(b)
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
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(LF, [operator], [FE], regions, NumberType, 1, bonus_quadorder, verbosity - 1)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE))
    cvals_resultdim = size(basisevaler[1][1].cvals[1],1)
    action_resultdim = action.resultdim
    @assert action_resultdim == 1

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item = 0 # number of dofs for item
    evalnr = [0] # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    maxdofs = max_num_targets_per_source(xItemDofs)
    dofs = zeros(Int32,maxdofs)
    temp = 0 # some temporary variable
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action_resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEAssemblyvalue at quadrature point i
    weights::Array{NumberType,1} = [] # pointer to quadrature weights
    localb = zeros(NumberType,maxdofs)
    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # item number for dof provider
        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        dofs[1:ndofs4item] = xItemDofs[:,dofitem]

        weights = qf[iEG].w
        for i in eachindex(weights)
            cvali = basisevaler[iEG][evalnr[1]].cvals[i]

            for dof_i = 1 : ndofs4item
                # apply action
                for k = 1 : cvals_resultdim
                    action_input[k] = cvali[k,dof_i]
                end    
                apply_action!(action_result, action_input, action, i)

                for j = 1 : action_resultdim
                    localb[dof_i] += action_result[j] * weights[i]
                end
            end 
        end  

        for dof_i = 1 : ndofs4item
            b[dofs[dof_i]] += localb[dof_i] * xItemVolumes[item]
        end
        fill!(localb, 0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end




function assemble!(
    A::AbstractArray{<:Real,2},
    BLF::BilinearForm{AT};
    verbosity::Int = 0,
    transpose_copy = Nothing) where AT <: AbstractAssemblyType

    FE = [BLF.FE1, BLF.FE2]
    operator = [BLF.operator1, BLF.operator2]
    action = BLF.action
    bonus_quadorder = BLF.action.bonus_quadorder
    regions = BLF.regions

    
    # collect grid information
    NumberType = eltype(A)
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
    EG, ndofs4EG, qf, basisevaler, EG4item, dofitem4item, evaler4item = prepareOperatorAssembly(BLF, operator, FE, regions, NumberType, 2, bonus_quadorder, verbosity-1)

    # collect FE and FEBasisEvaluator information
    ncomponents = FiniteElements.get_ncomponents(typeof(FE[1]))
    cvals_resultdim::Int = size(basisevaler[1][1].cvals[1],1)
    action_resultdim::Int = action.resultdim

    # loop over items
    itemET = xItemGeometries[1] # type of the current item
    iEG = 1 # index to the correct unique geometry
    ndofs4item1 = ndofs4EG[1] # number of dofs for item
    ndofs4item2 = ndofs4EG[1] # number of dofs for item
    evalnr = [0,0] # evaler number that has to be used for current item
    dofitem = 0 # itemnr where the dof numbers can be found
    maxdofs1 = max_num_targets_per_source(xItemDofs1)
    maxdofs2 = max_num_targets_per_source(xItemDofs2)
    dofs = zeros(Int,maxdofs1)
    dofs2 = zeros(Int,maxdofs2)
    temp::NumberType = 0 # some temporary variable
    action_input = zeros(NumberType,cvals_resultdim) # heap for action input
    action_result = zeros(NumberType,action_resultdim) # heap for action output
    cvali::Array{NumberType,2} = [[] []] # pointer to FEAssemblyvalue at quadrature point i
    cvali2::Array{NumberType,2} = [[] []] # pointer to FEAssemblyvalue at quadrature point i
    weights::Array{NumberType,1} = [] # pointer to quadrature weights
    localmatrix = zeros(NumberType,maxdofs1,maxdofs2)
    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        dofitem = dofitem4item(item)

        # find index for CellType
        itemET = EG4item(item)
        iEG = findfirst(isequal(itemET), EG)
        ndofs4item1 = ndofs4EG[1][iEG]
        ndofs4item2 = ndofs4EG[2][iEG]

        # update FEbasisevaler
        evalnr = evaler4item(item)
        update!(basisevaler[iEG][evalnr[1]],dofitem)
        update!(basisevaler[iEG][evalnr[2]],dofitem)

        # update action
        update!(action, basisevaler[iEG][evalnr[1]], item, regions[r])

        # update dofs
        dofs[1:ndofs4item1] = xItemDofs1[:,dofitem]
        dofs2[1:ndofs4item2] = xItemDofs2[:,dofitem]

        weights = qf[iEG].w
        for i in eachindex(weights)
            cvali = basisevaler[iEG][evalnr[1]].cvals[i]
            cvali2 = basisevaler[iEG][evalnr[2]].cvals[i]
           
            for dof_i = 1 : ndofs4item1
                # apply action to first argument
                for k = 1 : cvals_resultdim
                    action_input[k] = cvali[k,dof_i]
                end    
                apply_action!(action_result, action_input, action, i)

                if BLF.symmetric == false
                    for dof_j = 1 : ndofs4item2
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k]*cvali2[k,dof_j]
                        end
                        localmatrix[dof_i,dof_j] += temp * weights[i]
                    end
                else # symmetric case
                    for dof_j = dof_i : ndofs4item2
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k]*cvali2[k,dof_j]
                        end
                        localmatrix[dof_i,dof_j] += temp * weights[i]
                    end
                end
            end 
        end 

        # copy localmatrix into global matrix
        if BLF.symmetric == false
            for dof_i = 1 : ndofs4item1, dof_j = 1 : ndofs4item2
                A[dofs2[dof_j],dofs[dof_i]] += localmatrix[dof_i,dof_j] * xItemVolumes[item]    
                if transpose_copy != Nothing
                    transpose_copy[dofs[dof_i],dofs2[dof_j]] += localmatrix[dof_i,dof_j] * xItemVolumes[item]
                end
                localmatrix[dof_i,dof_j] = 0.0
            end
        else # symmetric case
            for dof_i = 1 : ndofs4item1, dof_j = dof_i+1 : ndofs4item2
                temp = localmatrix[dof_i,dof_j] * xItemVolumes[item]
                A[dofs2[dof_j],dofs[dof_i]] += temp
                A[dofs2[dof_i],dofs[dof_j]] += temp
                localmatrix[dof_i,dof_j] = 0.0
            end    
            for dof_i = 1 : ndofs4item1
                A[dofs2[dof_i],dofs[dof_i]] += localmatrix[dof_i,dof_i] * xItemVolumes[item]
                localmatrix[dof_i,dof_i] = 0.0
            end    
        end    

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
end
