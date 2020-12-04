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
$(TYPEDEF)

assembly pattern nonlinear form with two arguments (ansatz and testfunction) where
te first argument can depend on more than one operator
"""
struct NonlinearForm{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    FE1::Array{FESpace,1}    # finite element spaces for each operator of the ansatz function
    FE2::FESpace             # finite element space for testfunction
    operator1::Array{DataType,1}   # operators that should be evaluated for the ansatz function
    operator2::Type{<:AbstractFunctionOperator}   # operator that is evaluated for the test function
    action::AbstractAction  # is applied to all operators of ansatz functions (to allow for Newton)
                            # in Newton mode also the evaluation of all operators in current solution are
                            # passed to the action
    regions::Array{Int,1}
    use_newton::Bool        # performs automatic Newton derivatives of the action
                            # (all operators are expected to depend linearly on the coefficients)
end   


"""
````
function NonlinearForm(
    T::Type{<:Real},
    FE1::Array{FESpace,1},          # finite element spaces for each operator of the ansatz function
    FE2::FESpace,                   # finite element space for testfunction
    operator1::Array{DataType,1},   # operators that should be evaluated for the ansatz function
    operator2::Type{<:AbstractFunctionOperator},   # operator that is evaluated for the test function
    action::AbstractAction;        # is applied to all operators of ansatz functions (to allow for Newton)
                                   # in Newton mode also the evaluation of all operators in current solution are
                                   # passed to the action
    ADnewton::Bool = false,        # perform AD to obtain additional Newton terms
    regions::Array{Int,1} = [0])
````

Creates a NonlinearForm.
"""
function NonlinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE1::Array{FESpace,1},    # finite element spaces for each operator of the ansatz function
    FE2::FESpace,             # finite element space for testfunction
    operator1::Array{DataType,1},   # operators that should be evaluated for the ansatz function
    operator2::Type{<:AbstractFunctionOperator},   # operator that is evaluated for the test function
    action::AbstractAction;  # is applied to all operators of ansatz functions (to allow for Newton)
                             # in Newton mode also the evaluation of all operators in current solution are
                             # passed to the action
    ADnewton::Bool = false,  # perform AD to obtain additional Newton terms
    regions::Array{Int,1} = [0])
    return NonlinearForm{T,AT}(FE1,FE2,operator1,operator2,action,regions,false)
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

assembly pattern multi-linear form (that depends on arbitrary many quantities)

currently this can be only assembled into a FEVector (with one free argument, all others fixed)
"""
struct MultilinearForm{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    FE::Array{FESpace,1}
    operators::Array{DataType,1} # operators for each element of FE
    action::AbstractAction # is applied to all fixed arguments
    regions::Array{Int,1}
end   

"""
````
function MultilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
````

Creates a MultilinearForm that can be only assembled into a vector (with all but one fixed arguments).
"""
function MultilinearForm(
    T::Type{<:Real},
    AT::Type{<:AbstractAssemblyType},
    FE::Array{FESpace,1},
    operators::Array{DataType,1}, 
    action::AbstractAction; 
    regions::Array{Int,1} = [0])
    return MultilinearForm{T,AT}(FE,operators,action,regions)
end


"""
$(TYPEDEF)

assembly pattern item integrator that can e.g. be used for error/norm evaluations
"""
struct ItemIntegrator{T <: Real, AT <: AbstractAssemblyType} <: AbstractAssemblyPattern{T, AT}
    operators::Array{DataType,1}
    action::AbstractAction # is applied to all operators
    regions::Array{Int,1}
end

# single operator constructor
function ItemIntegrator{T, AT}(operator::Type{<:AbstractFunctionOperator}, action::AbstractAction, regions = [0]) where {T <: Real, AT <: AbstractAssemblyType}
    return ItemIntegrator{T,AT}([operator],action,regions)
end

function prepare_assembly(
    form::AbstractAssemblyPattern{T,AT},
    operator::Array{DataType,1},
    FE::Array{<:FESpace,1},
    regions::Array{Int32,1},
    nrfactors::Int,
    bonus_quadorder::Int,
    verbosity::Int) where {T<: Real, AT <: AbstractAssemblyType}

    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    EG = FE[1].xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]

    # note: EG are the element geometries related to AT (the real integration domains)

    # find out which operators need another assembly type
    # e.g. FaceJump operators that are assembled ON_CELLS
    # and get the corressponding dofmaps
    dofitemAT = Array{Type{<:AbstractAssemblyType},1}(undef,length(FE))
    facejump_operators = []
    for j=1:length(FE)
        dofitemAT[j] = DofitemAT4Operator(AT, operator[j])
        xItemDofs[j] = Dofmap4AssemblyType(FE[j],dofitemAT[j])
        if (dofitemAT[j] == ON_CELLS) && (AT <: ON_FACES || AT <: ON_BFACES)
            push!(facejump_operators,j)
        end
    end    

    # if one of the operators is a face jump operator we also need the element geometries 
    # of the neighbouring cells
    EGoffset = length(EG)
    EGdofitem = []
    if length(facejump_operators) > 0
        xDofItemGeometries = FE[facejump_operators[1]].xgrid[GridComponentGeometries4AssemblyType(dofitemAT[facejump_operators[1]])]
        xDofItemRegions = FE[facejump_operators[1]].xgrid[GridComponentRegions4AssemblyType(dofitemAT[facejump_operators[1]])]
        xDofItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(facejump_operators))
        for k = 1 : length(facejump_operators)
            xDofItemDofs[k] = Dofmap4AssemblyType(FE[facejump_operators[k]], dofitemAT[facejump_operators[k]])
        end

        # find unique ElementGeometries and coressponding ndofs
        EGdofitem = FE[1].xgrid[GridComponentUniqueGeometries4AssemblyType(dofitemAT[facejump_operators[1]])]
    end

    # get function that handles the dofitem information for every integration item
    dii4op = Array{Function,1}(undef,length(FE))
    for j=1:length(FE)
        dii4op[j]  = DofitemInformation4Operator(FE[j], EG, EGdofitem, AT, operator[j])
    end

    # find proper quadrature QuadratureRules
    # and construct matching FEBasisEvaluators
    # dimension 1 = id of element geometry (combination of integration domains/dofitem domains)
    # dimension 2 = id finite element
    # dimension 3 = position if integration domain in superset dofitem (if jumping operator)
    # dimension 4 = left or right side (if jumping operator)
    qf = Array{QuadratureRule,1}(undef,length(EG) + length(EGdofitem))
    quadorder = 0
    ndofs4EG = Array{Array{Int,1},1}(undef,length(FE))
    for e = 1 : length(FE)
        ndofs4EG[e] = zeros(Int,length(EG)+length(EGdofitem))
    end

    maxfaces = 0
    maxorientations = 0
    for j = 1 : length(EGdofitem)        
        maxfaces = max(maxfaces,nfaces_for_geometry(EGdofitem[j]))
    end
    if length(facejump_operators) > 0
        for j = 1 : length(EG)  
            maxorientations = max(maxorientations, length(xrefFACE2xrefOFACE(EG[j])))
        end
    end
    basisevaler = Array{FEBasisEvaluator,4}(undef, length(EG) + length(EGdofitem), length(FE), (length(facejump_operators) > 0) ? maxfaces : 1, (length(facejump_operators) > 0) ? maxorientations : 1)
    for j = 1 : length(EG)
        quadorder = bonus_quadorder
        for k = 1 : length(FE)
            quadorder += get_polynomialorder(eltype(FE[k]), EG[j]) + QuadratureOrderShift4Operator(eltype(FE[k]),operator[k])
        end
        quadorder = max(quadorder,0)          
        qf[j] = QuadratureRule{T,EG[j]}(quadorder);
        # choose quadrature order for all finite elements
        for k = 1 : length(FE)
          #  if dofitemAT[k] == AT
                if k > 1 && FE[k] == FE[1] && operator[k] == operator[1]
                    basisevaler[j,k,1,1] = basisevaler[j,1,1,1] # e.g. for symmetric bilinerforms
                elseif k > 2 && FE[k] == FE[2] && operator[k] == operator[2]
                    basisevaler[j,k,1,1] = basisevaler[j,2,1,1]
                else    
                    basisevaler[j,k,1,1] = FEBasisEvaluator{T,eltype(FE[k]),EG[j],operator[k],AT}(FE[k], qf[j]; verbosity = verbosity - 1)
                end    
                ndofs4EG[k][j] = size(basisevaler[j,k,1,1].cvals,2)
         #   end
        end
    end        

    # assign additional basisevaler for FaceJump operators: for each face
    # of each unique cell geometry (in EGdofitem) quadrature points of their face geometry are mapped to
    # quadrature points on the cell (NOTE: We assume that all faces of an EGdofitem are of the same shape)
    if length(facejump_operators) > 0
        for j = 1 : length(EGdofitem)
            quadorder = bonus_quadorder
            for k = 1 : length(FE)
                quadorder += get_polynomialorder(eltype(FE[k]), EGdofitem[j]) + QuadratureOrderShift4Operator(eltype(FE[k]),operator[k])
            end
            quadorder = max(quadorder,0)        
            nfaces4cell = nfaces_for_geometry(EGdofitem[j])
            EGface = facetype_of_cellface(EGdofitem[j], 1)
            EGfaceid = 0
            for f = 1 : length(EG)
                if EG[f] == EGface
                    EGfaceid = f
                    break;
                end
            end
            # load quadrature rule for face
            qf4face = qf[EGfaceid]

            # generate new quadrature rules on neighbouring cells
            # where quadrature points of face are mapped to quadrature points of cells
            qf[EGoffset + j] = QuadratureRule{T,EGdofitem[j]}(qf4face.name * " (shape faces)",Array{Array{T,1},1}(undef,length(qf4face.xref)),qf4face.w)
            for k in facejump_operators
                xrefFACE2CELL = xrefFACE2xrefCELL(EGdofitem[j])
                EGface = facetype_of_cellface(EGdofitem[j], 1)
                xrefFACE2OFACE = xrefFACE2xrefOFACE(EGface)
                for f = 1 : nfaces4cell, orientation = 1 : length(xrefFACE2OFACE)
                    for i = 1 : length(qf4face.xref)
                        qf[EGoffset + j].xref[i] = xrefFACE2CELL[f](xrefFACE2OFACE[orientation](qf4face.xref[i]))
                        #println("face $f orientation $orientation : mapping  $(qf4face.xref[i]) to $(qf[EGoffset + j].xref[i])")
                    end
                    basisevaler[EGoffset + j,k,f,orientation] = FEBasisEvaluator{T,eltype(FE[k]),EGdofitem[j],operator[k],dofitemAT[k]}(FE[k], qf[EGoffset + j]; verbosity = verbosity - 1)
                end
                ndofs4EG[k][EGoffset+j] = size(basisevaler[EGoffset + j,k,1,1].cvals,2)
            end
        end

        # append EGdofitem to EG
        EG = [EG, EGdofitem]
    end



    if verbosity > 0
        println("\nASSEMBLY PREPARATION $(typeof(form))")
        println("====================================")
        println("      action = $(typeof(form.action))")
        println("     regions = $regions")
        println("          EG = $EG")
        println("\n  List of arguments FEType / operator / ndofs4EG:")
        for k = 1 : length(FE)
            println("      ($k) $(FE[k].name) / $(operator[k]) / $(ndofs4EG[k])")
        end    
        if verbosity > 1
            for j = 1 : length(EG)
                println("\nQuadratureRule [$j] for $(EG[j]):")
                Base.show(qf[j])
            end
        end
    end

    return EG, ndofs4EG, qf, basisevaler, dii4op
end


function parse_regions(regions, xItemRegions)
    regions = deepcopy(regions)
    if regions == [0]
        try
            regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
        catch
            regions = [xItemRegions[1]]
        end        
    else
        regions = Array{Int32,1}(regions)    
    end
    return regions
end



"""
````
function evaluate!(
    b::AbstractArray{T,2},
    form::ItemIntegrator{T,AT},
    FEB::FEVectorBlock;
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}
````

Evaluation of an ItemIntegrator form with given FEVectorBlock FEB into given two-dimensional Array b.
"""
function evaluate!(
    b::AbstractArray{T,2},
    form::ItemIntegrator{T,AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    operators = form.operators
    @assert length(FEB) == length(form.operators)
    nFE = length(FEB)
    FE = Array{FESpace,1}(undef, nFE)
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef, nFE)
    for j = 1 : nFE
        FE[j] = FEB[j].FES
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, operators[j]))
    end
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = form.action
    regions = parse_regions(form.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(form, operators, FE, regions, nFE, action.bonus_quadorder, verbosity - 1)

    # get size informations
    ncomponents = zeros(Int,nFE)
    offsets = zeros(Int,nFE+1)
    maxdofs = 0
    for j = 1 : nFE
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[end,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]
    maxdofs2 = max_num_targets_per_source(xItemDofs[end])

    maxnweights = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end]) # heap for action input
    end
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output

    # loop over items
    EG4item = 0
    EG4dofitem = [1,1] # type of the current item
    ndofs4dofitem = 0 # number of dofs for item
    dofitems = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem::Array{T,1} = [0.0,0.0]
    dofitem = 0
    coeffs = zeros(T,maxdofs)
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::FEBasisEvaluator = basisevaler[1,1,1,1]

    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

        if dofitems[1] == 0
            break;
        end

        # get information on dofitems
        weights = qf[EG4item].w
        for di = 1 : length(dofitems)
            dofitem = dofitems[di]
            if dofitem != 0
                for FEid = 1 : nFE
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem[di],FEid,itempos4dofitem[di],orientation4dofitem[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem[di]]
                    for j=1:ndofs4dofitem
                        fdof = xItemDofs[FEid][j,dofitem]
                        coeffs[j] = FEB[FEid][fdof]
                    end

                    for i in eachindex(weights)
                        if FEid == 1 && di == 1
                            fill!(action_input[i], 0)
                        end
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i; offset = offsets[FEid], factor = coefficient4dofitem[di])
                    end  
                end
            end
        end

        # update action on item/dofitem
       # basisevaler4dofitem = basisevaler[EG4item[1],1,1,1]
        update!(action, basisevaler4dofitem, dofitems[1], item, regions[r])

        for i in eachindex(weights)
            # apply action to FEVector
            apply_action!(action_result, action_input[i], action, i)
            for j = 1 : action_resultdim
                b[j,item] += action_result[j] * weights[i] * xItemVolumes[item]
            end
        end  

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
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
    FEB;
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # quick and dirty : we mask the resulting array as an AbstractArray{T,2} using AccumulatingVector
    # and use the itemwise evaluation above
    resultdim = form.action.argsizes[1]
    AV = AccumulatingVector{T}(zeros(T,resultdim), 0)

    if typeof(FEB) <: Array{<:FEVectorBlock,1}
        evaluate!(AV, form, FEB; verbosity = verbosity)
    else
        evaluate!(AV, form, [FEB]; verbosity = verbosity)
    end

    if resultdim == 1
        return AV.entries[1]
    else
        return AV.entries
    end
end


"""
````
assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},
    LF::LinearForm{T,AT};
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

````

Assembly of a LinearForm LF into given one- or two-dimensional Array b.
"""
function assemble!(
    b::Union{AbstractArray{T,1},AbstractArray{T,2}},
    LF::LinearForm{T,AT};
    verbosity::Int = 0,
    factor = 1,
    offset = 0) where {T <: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = LF.FE
    operator = LF.operator
    xItemVolumes::Array{T,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE, DofitemAT4Operator(AT, operator))
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = LF.action
    regions::Array{Int32,1} = parse_regions(LF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(LF, [operator], [FE], regions, 1, action.bonus_quadorder, verbosity - 1)

    # get size informations
    ncomponents::Int = get_ncomponents(eltype(FE))
    cvals_resultdim::Int = size(basisevaler[1,1,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    if typeof(b) <: AbstractArray{T,1}
        @assert action_resultdim == 1
        onedimensional = true
    else
        onedimensional = false
    end

    # loop over items
    EG4dofitem = [1,1] # type of the current item
    ndofs4dofitem = 0 # number of dofs for item
    dofitems = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem = [0,0]
    dofitem = 0
    maxndofs = max_num_targets_per_source(xItemDofs)
    dofs = zeros(Int,maxndofs)
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1]
    localb::Array{T,2} = zeros(T,maxndofs,action_resultdim)
    bdof::Int = 0
    itemfactor::T = 0

    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations
        dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        itemfactor = factor * xItemVolumes[item]

        # loop over associated dofitems
        for di = 1 : length(dofitems)
            dofitem = dofitems[di]
            if dofitem != 0

                # get number of dofs on this dofitem
                ndofs4dofitem = ndofs4EG[1][EG4dofitem[di]]

                # update FEbasisevaler on dofitem
                basisevaler4dofitem = basisevaler[EG4dofitem[di],1,itempos4dofitem[di],orientation4dofitem[di]]
                update!(basisevaler4dofitem,dofitem)

                # update action on dofitem
                update!(action, basisevaler4dofitem, dofitem, item, regions[r])

                # update dofs
                for j=1:ndofs4dofitem
                    dofs[j] = xItemDofs[j,dofitem]
                end

                weights = qf[EG4dofitem[di]].w
                for i in eachindex(weights)
                    for dof_i = 1 : ndofs4dofitem
                        # apply action
                        eval!(action_input, basisevaler4dofitem , dof_i, i)
                        apply_action!(action_result, action_input, action, i)
                        for j = 1 : action_resultdim
                            localb[dof_i,j] += action_result[j] * weights[i]
                        end
                    end 
                end  

                if onedimensional
                    for dof_i = 1 : ndofs4dofitem
                        bdof = dofs[dof_i] + offset
                        b[bdof] += localb[dof_i,1] * itemfactor
                    end
                else
                    for dof_i = 1 : ndofs4dofitem, j = 1 : action_resultdim
                        bdof = dofs[dof_i] + offset
                        b[bdof,j] += localb[dof_i,j] * itemfactor
                    end
                end
                fill!(localb, 0.0)
            end
        end
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end

function assemble!(
    b::FEVectorBlock,
    LF::LinearForm;
    verbosity::Int = 0,
    factor = 1)

    assemble!(b.entries, LF; verbosity = verbosity, factor = factor, offset = b.offset)
end


"""
````
assemble!(
    A::AbstractArray{T,2},
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    transpose_copy = Nothing) where {T<: Real, AT <: AbstractAssemblyType}
````

Assembly of a BilinearForm BLF into given two-dimensional AbstractArray (e.g. FEMatrixBlock).
"""
function assemble!(
    A::AbstractArray{T,2},
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    transpose_copy = nothing,
    offsetX = 0,
    offsetY = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = [BLF.FE1, BLF.FE2]
    operator = [BLF.operator1, BLF.operator2]
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs1::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[1], DofitemAT4Operator(AT, operator[1]))
    xItemDofs2::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[2], DofitemAT4Operator(AT, operator[2]))
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = BLF.action
    regions = parse_regions(BLF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(BLF, operator, FE, regions, 2, action.bonus_quadorder, verbosity - 1)
 
    # get size informations
    ncomponents::Int = get_ncomponents(eltype(FE[1]))
    cvals_resultdim::Int = size(basisevaler[end,apply_action_to,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]
 
    # loop over items
    EG4item::Int = 1
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0.0,0.0] # coefficients for operator 1
    coefficient4dofitem2::Array{T,1} = [0.0,0.0] # coefficients for operator 2
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    dofitem1 = 0
    dofitem2 = 0
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    dofs = zeros(Int,maxdofs1)
    dofs2 = zeros(Int,maxdofs2)
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem1::FEBasisEvaluator = basisevaler[1,1,1,1]
    basisevaler4dofitem2::FEBasisEvaluator = basisevaler[1,2,1,1]
    basisvals1::Array{T,3} = basisevaler4dofitem1.cvals
    basisvals2::Array{T,3} = basisevaler4dofitem2.cvals
    localmatrix::Array{T,2} = zeros(T,maxdofs1,maxdofs2)
    temp::T = 0 # some temporary variable
    acol::Int = 0
    arow::Int = 0
    is_locally_symmetric::Bool = BLF.symmetric
    itemfactor::T = 0
    
    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)

        # get quadrature weights for integration domain
        weights = qf[EG4item].w
        itemfactor = xItemVolumes[item] * factor

        # loop over associated dofitems (maximal 2 for jump calculations)
        # di, dj == 2 is only performed if one of the operators jumps
        for di = 1 : 2, dj = 1 : 2
            dofitem1 = dofitems1[di]
            dofitem2 = dofitems2[dj]
            if dofitem1 > 0 && dofitem2 > 0

                # even if global matrix is symmeric, local matrix might be not in case of JumpOperators
                is_locally_symmetric = BLF.symmetric * (dofitem1 == dofitem2)

                # get number of dofs on this dofitem
                ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]

                # update FEbasisevaler
                basisevaler4dofitem1 = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisvals1 = basisevaler4dofitem1.cvals
                basisvals2 = basisevaler4dofitem2.cvals
                update!(basisevaler4dofitem1,dofitem1)
                update!(basisevaler4dofitem2,dofitem2)

                # update action on dofitem
                if apply_action_to == 1
                    update!(action, basisevaler4dofitem1, dofitem1, item, regions[r])
                else
                    update!(action, basisevaler4dofitem2, dofitem2, item, regions[r])
                end

                # update dofs
                for j=1:ndofs4item1
                    dofs[j] = xItemDofs1[j,dofitem1]
                end
                for j=1:ndofs4item2
                    dofs2[j] = xItemDofs2[j,dofitem2]
                end

                for i in eachindex(weights)
                    if apply_action_to == 1
                        for dof_i = 1 : ndofs4item1
                            eval!(action_input, basisevaler4dofitem1, dof_i, i)
                            apply_action!(action_result, action_input, action, i)
                            action_result .*= coefficient4dofitem1[di]

                            if is_locally_symmetric == false
                                for dof_j = 1 : ndofs4item2
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals2[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem2[dj]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            else # symmetric case
                                for dof_j = dof_i : ndofs4item2
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals2[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem2[dj]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            end
                        end 
                    else
                        for dof_j = 1 : ndofs4item2
                            eval!(action_input, basisevaler4dofitem2, dof_j, i)
                            apply_action!(action_result, action_input, action, i)
                            action_result .*= coefficient4dofitem2[dj]

                            if is_locally_symmetric == false
                                for dof_i = 1 : ndofs4item1
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals1[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem1[di]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            else # symmetric case
                                for dof_i = dof_j : ndofs4item1
                                    temp = 0
                                    for k = 1 : action_resultdim
                                        temp += action_result[k] * basisvals1[k,dof_j,i]
                                    end
                                    temp *= coefficient4dofitem1[di]
                                    localmatrix[dof_i,dof_j] += weights[i] * temp
                                end
                            end
                        end 
                    end
                end

                # copy localmatrix into global matrix
                if is_locally_symmetric == false
                    for dof_i = 1 : ndofs4item1
                        arow = dofs[dof_i] + offsetX
                        for dof_j = 1 : ndofs4item2
                            if localmatrix[dof_i,dof_j] != 0
                                acol = dofs2[dof_j] + offsetY
                                if transposed_assembly == true
                                    _addnz(A,acol,arow,localmatrix[dof_i,dof_j] * itemfactor,1)
                                else 
                                    _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)  
                                end
                                if transpose_copy != nothing # sign is changed in case nonzero rhs data is applied to LagrangeMultiplier (good idea?)
                                    if transposed_assembly == true
                                        _addnz(transpose_copy,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,-1)
                                    else
                                        _addnz(transpose_copy,acol,arow,localmatrix[dof_i,dof_j] * itemfactor,-1)
                                    end
                                end
                            end
                        end
                    end
                else # symmetric case
                    for dof_i = 1 : ndofs4item1
                        for dof_j = dof_i+1 : ndofs4item2
                            if localmatrix[dof_i,dof_j] != 0 
                                arow = dofs[dof_i] + offsetX
                                acol = dofs2[dof_j] + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                                arow = dofs[dof_j] + offsetX
                                acol = dofs2[dof_i] + offsetY
                                _addnz(A,arow,acol,localmatrix[dof_i,dof_j] * itemfactor,1)
                            end
                        end
                    end    
                    for dof_i = 1 : ndofs4item1
                        arow = dofs2[dof_i] + offsetX
                        acol = dofs[dof_i] + offsetY
                       _addnz(A,arow,acol,localmatrix[dof_i,dof_i] * itemfactor,1)
                    end    
                end    
                fill!(localmatrix,0.0)
            end
        end 
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end


## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    A::FEMatrixBlock,
    BLF::BilinearForm;
    apply_action_to::Int = 1,
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    transpose_copy = nothing)

    if typeof(transpose_copy) <: FEMatrixBlock
        assemble!(A.entries, BLF; apply_action_to = apply_action_to, verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, transpose_copy = transpose_copy.entries, offsetX = A.offsetX, offsetY = A.offsetY)
    else
        assemble!(A.entries, BLF; apply_action_to = apply_action_to, verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, transpose_copy = transpose_copy, offsetX = A.offsetX, offsetY = A.offsetY)
    end
end



"""
````
assemble!(
    b::AbstractArray{T,1},
    fixedFE::FEVectorBlock,    # coefficient for fixed 2nd component
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    factor = 1,
    verbosity::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}
````

Assembly of a BilinearForm BLF into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the second argument is fixed by the given coefficients in fixedFE.
With apply_action_to=2 the action can be also applied to the second fixed argument instead of the first one (default).
"""
function assemble!(
    b::AbstractArray{T,1},
    fixedFE::AbstractArray{T,1},    # coefficient for fixed 2nd component
    BLF::BilinearForm{T, AT};
    apply_action_to::Int = 1,
    factor = 1,
    verbosity::Int = 0,
    offset = 0,
    offset2 = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = [BLF.FE1, BLF.FE2]
    operator = [BLF.operator1, BLF.operator2]
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs1::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[1], DofitemAT4Operator(AT, operator[1]))
    xItemDofs2::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[2], DofitemAT4Operator(AT, operator[2]))
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = BLF.action
    regions = parse_regions(BLF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(BLF, operator, FE, regions, 2, action.bonus_quadorder, verbosity - 1)


    # get size informations
    ncomponents::Int = get_ncomponents(eltype(FE[1]))
    ncomponents2::Int = get_ncomponents(eltype(FE[2]))
    cvals_resultdim::Int = size(basisevaler[1,apply_action_to,1,1].cvals,1)
    cvals_resultdim2::Int = size(basisevaler[1,2,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    # loop over items
    EG4dofitem1 = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2 = [1,1] # EG id of the current item with respect to operator 2
    dofitems1 = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2 = [0,0] # itemnr where the dof numbers can be found (operator 2)
    itempos4dofitem1 = [1,1] # local item position in dofitem1
    itempos4dofitem2 = [1,1] # local item position in dofitem2
    orientation4dofitem1 = [1,2] # local orientation
    orientation4dofitem2 = [1,2] # local orientation
    coefficient4dofitem1 = [0,0] # coefficients for operator 1
    coefficient4dofitem2 = [0,0] # coefficients for operator 2
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    dofitem1 = 0
    dofitem2 = 0
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    dofs = zeros(Int,maxdofs1)
    coeffs2 = zeros(T,maxdofs2)
    action_input::Array{T,1} = zeros(T,cvals_resultdim) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem1 = basisevaler[1,1,1,1]
    basisevaler4dofitem2 = basisevaler[1,2,1,1]
    basisvals1::Array{T,3} = basisevaler4dofitem1.cvals
    localmatrix = zeros(T,maxdofs1,maxdofs2)
    fixedval = zeros(T, cvals_resultdim2) # some temporary variable
    temp::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,maxdofs1)
    bdof::Int = 0
    fdof::Int = 0

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations
        dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)

        # get quadrature weights for integration domain
        weights = qf[EG4dofitem1[1]].w

        # loop over associated dofitems (maximal 2 for jump calculations)
        # di, dj == 2 is only performed if one of the operators jumps
        for di = 1 : 2, dj = 1 : 2
            dofitem1 = dofitems1[di]
            dofitem2 = dofitems2[dj]
            if dofitem1 > 0 && dofitem2 > 0

                # get number of dofs on this dofitem
                ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]

                # update FEbasisevaler
                basisevaler4dofitem1 = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisvals1 = basisevaler4dofitem1.cvals
                update!(basisevaler4dofitem1,dofitem1)
                update!(basisevaler4dofitem2,dofitem2)

                # update action on dofitem
                if apply_action_to == 1
                    update!(action, basisevaler4dofitem1, dofitem1, item, regions[r])
                else
                    update!(action, basisevaler4dofitem2, dofitem2, item, regions[r])
                end

                # update dofs
                for j=1:ndofs4item1
                    dofs[j] = xItemDofs1[j,dofitem1]
                end
                for j=1:ndofs4item2
                    fdof = xItemDofs2[j,dofitem2] + offset2
                    coeffs2[j] = fixedFE[fdof]
                end

                for i in eachindex(weights)
                
                    # evaluate second component
                    fill!(fixedval, 0.0)
                    eval!(fixedval, basisevaler4dofitem2, coeffs2, i)
                    fixedval .*= coefficient4dofitem2[dj]

                    if apply_action_to == 2
                        # apply action to second argument
                        apply_action!(action_result, fixedval, action, i)

                        # multiply first argument
                        for dof_i = 1 : ndofs4item1
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * basisvals1[k,dof_i,i]
                            end
                            localb[dof_i] += temp * weights[i] * coefficient4dofitem1[di]
                        end 
                    else
                        for dof_i = 1 : ndofs4item1
                            # apply action to first argument
                            eval!(action_input, basisevaler4dofitem1, dof_i, i)
                            apply_action!(action_result, action_input, action, i)
            
                            # multiply second argument
                            temp = 0
                            for k = 1 : action_resultdim
                                temp += action_result[k] * fixedval[k]
                            end
                            localb[dof_i] += temp * weights[i] * coefficient4dofitem1[di]
                        end 
                    end
                end

                for dof_i = 1 : ndofs4item1
                    bdof = dofs[dof_i] + offset
                    b[bdof] += localb[dof_i] * xItemVolumes[item] * factor
                end
                fill!(localb, 0.0)
            end
        end 
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end


# wrapper for FEVectorBlock to avoid setindex! functions of FEVectorBlock
function assemble!(
    b::FEVectorBlock,
    fixedFE::FEVectorBlock,    # coefficient for fixed 2nd component
    BLF::BilinearForm;
    apply_action_to::Int = 1,
    factor = 1,
    verbosity::Int = 0)

    assemble!(b.entries, fixedFE.entries, BLF; apply_action_to = apply_action_to, factor = factor, verbosity = verbosity, offset = b.offset, offset2 = fixedFE.offset)
end

"""
````
assemble!(
    assemble!(
    A::AbstractArray{T,2},
    FE1::FEVectorBlock,
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    fixed_argument::Int = 1,
    transposed_assembly::Bool = false,
    factor = 1)
````

Assembly of a TrilinearForm TLF into given two-dimensional AbstractArray (e.g. a FEMatrixBlock).
Here, the first argument is fixed by the given coefficients in FE1.
"""
function assemble!(
    A::AbstractArray{T,2},
    FE1::FEVectorBlock,
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    fixed_argument::Int = 1,
    transposed_assembly::Bool = false,
    factor = 1) where {T<: Real, AT <: AbstractAssemblyType}

    
    # get adjacencies
    FE = [TLF.FE1, TLF.FE2, TLF.FE3]
    operator = [TLF.operator1, TLF.operator2, TLF.operator3]
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1} = [Dofmap4AssemblyType(FE[1], DofitemAT4Operator(AT, operator[1])),
                 Dofmap4AssemblyType(FE[2], DofitemAT4Operator(AT, operator[2])),
                 Dofmap4AssemblyType(FE[3], DofitemAT4Operator(AT, operator[3]))]
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = TLF.action
    regions = parse_regions(TLF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(TLF, operator, FE, regions, 3, action.bonus_quadorder, verbosity - 1)


    # get size informations
    ncomponents = zeros(Int,length(FE))
    maxdofs = 0
    for j = 1 : length(FE)
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
    end
    action_resultdim::Int = action.argsizes[1]

    # loop over items
    EG4item::Int = 1
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    EG4dofitem3::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 3
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    dofitems3::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 3)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    itempos4dofitem3::Array{Int,1} = [1,1] # local item position in dofitem3
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem3::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0.0,0.0] # coefficients for operator 1
    coefficient4dofitem2::Array{T,1} = [0.0,0.0] # coefficients for operator 2
    coefficient4dofitem3::Array{T,1} = [0.0,0.0] # coefficients for operator 3
    ndofs4item::Array{Int, 1} = [0,0,0]
    dofitem1::Int = 0
    dofitem2::Int = 0
    dofitem3::Int = 0
    offsets = [0, size(basisevaler[1,1,1,1].cvals,1), size(basisevaler[1,1,1,1].cvals,1) + size(basisevaler[1,2,1,1].cvals,1)]
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::Array{FEBasisEvaluator,1} = [basisevaler[1,1,1,1], basisevaler[1,2,1,1], basisevaler[1,3,1,1]]
    basisvals_testfunction::Array{T,3} = basisevaler4dofitem[3].cvals
    evalfixedFE::Array{T,1} = zeros(T,size(basisevaler[1,fixed_argument,1,1].cvals,1)) # evaluation of argument 1
    temp::T = 0 # some temporary variable

    nonfixed_ids = setdiff([1,2,3], fixed_argument)
    coeffs::Array{T,1} = zeros(T,max_num_targets_per_source(xItemDofs[fixed_argument]))
    dofs2::Array{Int,1} = zeros(Int,max_num_targets_per_source(xItemDofs[nonfixed_ids[1]]))
    dofs3::Array{Int,1} = zeros(Int,max_num_targets_per_source(xItemDofs[nonfixed_ids[2]]))
    localmatrix::Array{T,2} = zeros(T,length(dofs2),length(dofs3))

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)
        dii4op[3](dofitems3, EG4dofitem3, itempos4dofitem3, coefficient4dofitem3, orientation4dofitem3, item)

        # get quadrature weights for integration domain
        weights = qf[EG4item].w

        # loop over associated dofitems (maximal 2 for jump calculations)
        for di = 1 : 2, dj = 1 : 2, dk = 1 : 2
            dofitem1 = dofitems1[di]
            dofitem2 = dofitems2[dj]
            dofitem3 = dofitems3[dk]

            if dofitem1 > 0 && dofitem2 > 0 && dofitem3 > 0

                # get number of dofs on this dofitem
                ndofs4item[1] = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item[2] = ndofs4EG[2][EG4dofitem2[dj]]
                ndofs4item[3] = ndofs4EG[3][EG4dofitem3[dk]]

                # update FEbasisevaler
                basisevaler4dofitem[1] = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem[2] = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisevaler4dofitem[3] = basisevaler[EG4dofitem3[dk],3,itempos4dofitem3[dk],orientation4dofitem3[dk]]
                update!(basisevaler4dofitem[1],dofitem1)
                update!(basisevaler4dofitem[2],dofitem2)
                update!(basisevaler4dofitem[3],dofitem3)

                # update action on dofitem
                update!(action, basisevaler4dofitem[fixed_argument], dofitem2, item, regions[r])

                # update coeffs of fixed argument
                for j=1:ndofs4item[fixed_argument]
                    coeffs[j] = FE1[xItemDofs[fixed_argument][j,dofitem1]]
                end
                # update dofs of free arguments
                for j=1:ndofs4item[nonfixed_ids[1]]
                    dofs2[j] = xItemDofs[nonfixed_ids[1]][j,dofitem2]
                end
                for j=1:ndofs4item[nonfixed_ids[2]]
                    dofs3[j] = xItemDofs[nonfixed_ids[2]][j,dofitem3]
                end

                if fixed_argument in [1,2]
                    basisvals_testfunction = basisevaler4dofitem[nonfixed_ids[2]].cvals
                    for i in eachindex(weights)
    
                        # evaluate fixed argument into action
                        fill!(action_input, 0.0)
                        eval!(action_input, basisevaler4dofitem[fixed_argument], coeffs, i; offset = offsets[fixed_argument], factor = coefficient4dofitem1[di])
                        
                        for dof_i = 1 : ndofs4item[nonfixed_ids[1]]
                            # apply action to fixed argument and first non-fixed argument
                            eval!(action_input, basisevaler4dofitem[nonfixed_ids[1]], dof_i, i, offset = offsets[nonfixed_ids[1]], factor = coefficient4dofitem2[dj])
                            
                            apply_action!(action_result, action_input, action, i)
            
                            for dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * basisvals_testfunction[k,dof_j,i]
                                end
                                localmatrix[dof_i,dof_j] += temp * weights[i] * coefficient4dofitem3[dk]
                            end
                        end 
                    end
                else # fixed argument is the last one
                    for i in eachindex(weights)
    
                        # evaluate fixed argument into separate vector
                        fill!(evalfixedFE, 0.0)
                        eval!(evalfixedFE, basisevaler4dofitem[fixed_argument], coeffs, i; factor = coefficient4dofitem3[di])
                        
                        for dof_i = 1 : ndofs4item[nonfixed_ids[1]]
                            # apply action to fixed argument and first non-fixed argument
                            eval!(action_input, basisevaler4dofitem[nonfixed_ids[1]], dof_i, i; factor = coefficient4dofitem1[di])
                            
                            for dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                                eval!(action_input, basisevaler4dofitem[nonfixed_ids[2]], dof_j, i; offset = offsets[2], factor = coefficient4dofitem2[dj])
                                apply_action!(action_result, action_input, action, i)
            
                                temp = 0
                                for k = 1 : action_resultdim
                                    temp += action_result[k] * evalfixedFE[k]
                                end
                                localmatrix[dof_i,dof_j] += temp * weights[i] * coefficient4dofitem3[dk]
                            end
                        end 
                    end
                end 
        
                # copy localmatrix into global matrix
                for dof_i = 1 : ndofs4item[nonfixed_ids[1]], dof_j = 1 : ndofs4item[nonfixed_ids[2]]
                    if localmatrix[dof_i,dof_j] != 0
                        if transposed_assembly == true
                            _addnz(A,dofs3[dof_j],dofs2[dof_i],localmatrix[dof_i,dof_j] * xItemVolumes[item], factor)
                        else
                            _addnz(A,dofs2[dof_i],dofs3[dof_j],localmatrix[dof_i,dof_j] * xItemVolumes[item], factor)
                        end
                    end
                end
                fill!(localmatrix,0.0)
            end
        end 
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end


"""
````
assemble!(
    assemble!(
    b::AbstractVector,
    FE1::FEVectorBlock,
    FE2::FEVectorBlock.
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    factor = 1)
````

Assembly of a TrilinearForm TLF into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the first two arguments are fixed by the given coefficients in FE1 and FE2.
"""
function assemble!(
    b::AbstractVector,
    FE1::FEVectorBlock,
    FE2::FEVectorBlock,
    TLF::TrilinearForm{T, AT};
    verbosity::Int = 0,
    factor::Real = 1,
    offset::Int = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = [TLF.FE1, TLF.FE2, TLF.FE3]
    operator = [TLF.operator1, TLF.operator2, TLF.operator3]
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs1::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[1], DofitemAT4Operator(AT, operator[1]))
    xItemDofs2::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[2], DofitemAT4Operator(AT, operator[2]))
    xItemDofs3::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE[3], DofitemAT4Operator(AT, operator[3]))
    xItemRegions::Array{Int,1} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = TLF.action
    regions = parse_regions(TLF.regions, xItemRegions)
    EG, ndofs4EG::Array{Array{Int,1},1}, qf, basisevaler, dii4op = prepare_assembly(TLF, operator, FE, regions, 3, action.bonus_quadorder, verbosity - 1)

    # get size informations
    ncomponents::Int = get_ncomponents(eltype(FE[1]))
    ncomponents2::Int = get_ncomponents(eltype(FE[2]))
    cvals_resultdim::Int = size(basisevaler[1,1,1,1].cvals,1)
    cvals_resultdim2::Int = size(basisevaler[1,2,1,1].cvals,1)
    action_resultdim::Int = action.argsizes[1]

    # loop over items
    EG4item::Int = 1
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 1
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 2
    EG4dofitem3::Array{Int,1} = [1,1] # EG id of the current item with respect to operator 3
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 1)
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 2)
    dofitems3::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found (operator 3)
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem1
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem2
    itempos4dofitem3::Array{Int,1} = [1,1] # local item position in dofitem3
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem3::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{T,1} = [0.0,0.0] # coefficients for operator 1
    coefficient4dofitem2::Array{T,1} = [0.0,0.0] # coefficients for operator 2
    coefficient4dofitem3::Array{T,1} = [0.0,0.0] # coefficients for operator 3
    ndofs4item1::Int = 0 # number of dofs for item
    ndofs4item2::Int = 0 # number of dofs for item
    ndofs4item3::Int = 0 # number of dofs for item
    dofitem1::Int = 0
    dofitem2::Int = 0
    dofitem3::Int = 0
    maxdofs1::Int = max_num_targets_per_source(xItemDofs1)
    maxdofs2::Int = max_num_targets_per_source(xItemDofs2)
    maxdofs3::Int = max_num_targets_per_source(xItemDofs3)
    coeffs1::Array{T,1} = zeros(T,maxdofs1)
    coeffs2::Array{T,1} = zeros(T,maxdofs2)
    dofs3::Array{Int,1} = zeros(Int,maxdofs3)
    action_input::Array{T,1} = zeros(T,action.argsizes[2]) # heap for action input
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem1::FEBasisEvaluator = basisevaler[1,1,1,1]
    basisevaler4dofitem2::FEBasisEvaluator = basisevaler[1,2,1,1]
    basisevaler4dofitem3::FEBasisEvaluator = basisevaler[1,3,1,1]
    basisvals3::Array{T,3} = basisevaler4dofitem3.cvals
    temp::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,maxdofs3)
    bdof::Int = 0

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)
        dii4op[2](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)
        dii4op[3](dofitems3, EG4dofitem3, itempos4dofitem3, coefficient4dofitem3, orientation4dofitem3, item)

        # get quadrature weights for integration domain
        weights = qf[EG4item].w

        # loop over associated dofitems (maximal 2 for jump calculations)
        # di, dj == 2 is only performed if one of the operators jumps
        for di = 1 : 2, dj = 1 : 2, dk = 1 : 2
            dofitem1 = dofitems1[di]
            dofitem2 = dofitems2[dj]
            dofitem3 = dofitems3[dk]

            if dofitem1 > 0 && dofitem2 > 0 && dofitem3 > 0

               # println("di/dj/dk = $di/$dj/$dk")

                # get number of dofs on this dofitem
                ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
                ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]
                ndofs4item3 = ndofs4EG[3][EG4dofitem3[dk]]

                # update FEbasisevaler
                basisevaler4dofitem1 = basisevaler[EG4dofitem1[di],1,itempos4dofitem1[di],orientation4dofitem1[di]]
                basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],2,itempos4dofitem2[dj],orientation4dofitem2[dj]]
                basisevaler4dofitem3 = basisevaler[EG4dofitem3[dk],3,itempos4dofitem3[dk],orientation4dofitem3[dk]]
                basisvals3 = basisevaler4dofitem3.cvals
                update!(basisevaler4dofitem1,dofitem1)
                update!(basisevaler4dofitem2,dofitem2)
                update!(basisevaler4dofitem3,dofitem3)

                # update action on dofitem
                update!(action, basisevaler4dofitem2, dofitem2, item, regions[r])

                # update coeffs, dofs
                for j=1:ndofs4item1
                    bdof = xItemDofs1[j,dofitem1]
                    coeffs1[j] = FE1[bdof]
                end
                for j=1:ndofs4item2
                    bdof = xItemDofs2[j,dofitem2]
                    coeffs2[j] = FE2[bdof]
                end
                for j=1:ndofs4item3
                    dofs3[j] = xItemDofs3[j,dofitem3]
                end

                for i in eachindex(weights)

                    # evaluate first and second component
                    fill!(action_input, 0.0)
                    eval!(action_input, basisevaler4dofitem1, coeffs1, i; factor = coefficient4dofitem1[di])
                    eval!(action_input, basisevaler4dofitem2, coeffs2, i; offset = cvals_resultdim, factor = coefficient4dofitem2[dj])
        
                    # apply action to FE1 and FE2
                    
                    apply_action!(action_result, action_input, action, i)
                   
                    # multiply third component
                    for dof_j = 1 : ndofs4item3
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals3[k,dof_j,i]
                        end
                        localb[dof_j] += temp * weights[i]
                    end 
                end 
        
                for dof_i = 1 : ndofs4item3
                    bdof = dofs3[dof_i] + offset
                    b[bdof] += localb[dof_i] * xItemVolumes[item] * factor * coefficient4dofitem3[dk]
                end
                fill!(localb, 0.0)
            end
        end 
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end


## wrapper for FEVectorBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    b::FEVectorBlock,
    FE1::FEVectorBlock,
    FE2::FEVectorBlock,
    TLF::TrilinearForm;
    verbosity::Int = 0,
    factor::Real = 1)

    assemble!(b.entries, FE1, FE2, TLF; verbosity = verbosity, factor = factor, offset = b.offset)
end



"""
````
assemble!(
    assemble!(
    b::AbstractVector,
    FE::Array{<:FEVectorBlock,1},
    MLF::MultilinearForm{T, AT};
    verbosity::Int = 0,
    factor = 1)
````

Assembly of a MultilinearForm MLF into given one-dimensional AbstractArray (e.g. a FEVectorBlock).
Here, the all but the last arguments are fixed by the given coefficients in the components of FE.
"""
function assemble!(
    b::AbstractVector,
    FEB::Array{AbstractVector,1},
    MLF::MultilinearForm{T, AT};
    verbosity::Int = 0,
    factor = 1,
    offset = 0,
    offsets2 = [0]) where {T<: Real, AT <: AbstractAssemblyType}

    # get adjacencies
    FE = MLF.FE
    operator = MLF.operators
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    for j = 1 : length(FE)
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, operator[j]))
    end
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = MLF.action
    regions = parse_regions(MLF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(MLF, operator, FE, regions, length(FE), action.bonus_quadorder, verbosity - 1)

    # get size informations
    ncomponents = zeros(Int,length(FE))
    offsets = zeros(Int,length(FE)+1)
    maxdofs = 0
    for j = 1 : length(FE)
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[1,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]

    maxnweights = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end]) # heap for action input
    end

    # loop over items
    EG4item::Int = 0
    EG4dofitem::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    coefficient4dofitem::Array{T,1} = [0,0] # coefficients for operator
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitem::Int = 0
    coeffs::Array{T,1} = zeros(T,maxdofs)
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1,1,1,1]
    basisvals::Array{T,3} = basisevaler4dofitem.cvals
    temp::T = 0 # some temporary variable
    localb::Array{T,1} = zeros(T,maxdofs)
    nFE::Int = length(FE)
    bdof::Int = 0
    fdof::Int = 0

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        for FEid = 1 : nFE - 1
            # get dofitem informations
            EG4item = dii4op[FEid](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

            # get information on dofitems
            weights = qf[EG4item].w
            for di = 1 : length(dofitems)
                dofitem = dofitems[di]
                if dofitem != 0
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem[di],FEid,itempos4dofitem[di],orientation4dofitem[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem[di]]
                    for j=1:ndofs4dofitem
                        fdof = xItemDofs[FEid][j,dofitem] + offsets2[FEid]
                        coeffs[j] = FEB[FEid][fdof]
                    end

                    for i in eachindex(weights)
                        if FEid == 1 && di == 1
                            fill!(action_input[i], 0)
                        end
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i; offset = offsets[FEid], factor = coefficient4dofitem[di])
                    end  
                end
            end
        end

        # update action on item/dofitem
        # beware: currently last operator must not be a FaceJump operator
        EG4item = dii4op[nFE](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)
        basisvals = basisevaler4dofitem.cvals
        ndofs4dofitem = ndofs4EG[nFE][EG4item]
        
        for di = 1 : length(dofitems)
            dofitem = dofitems[di]
            if dofitem != 0
                basisevaler4dofitem = basisevaler[EG4dofitem[di],nFE,itempos4dofitem[di],orientation4dofitem[di]]
                ndofs4dofitem = ndofs4EG[nFE][EG4dofitem[di]]
                update!(action, basisevaler4dofitem, dofitems[di], item, regions[r])

                for i in eachindex(weights)
        
                    # apply action
                    apply_action!(action_result, action_input[i], action, i)
        
                    # multiply third component
                    for dof_j = 1 : ndofs4dofitem
                        temp = 0
                        for k = 1 : action_resultdim
                            temp += action_result[k] * basisvals[k,dof_j,i]
                        end
                        localb[dof_j] += temp * weights[i] * coefficient4dofitem[di]
                    end 
                end  
        
                for dof_i = 1 : ndofs4dofitem
                    bdof = xItemDofs[nFE][dof_i,dofitem] + offset
                    b[bdof] += localb[dof_i] * xItemVolumes[item] * factor
                end
            end
        end
        
        fill!(localb, 0.0)

        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop
    return nothing
end


function assemble!(
    b::FEVectorBlock,
    FEB::Array{<:FEVectorBlock,1},
    MLF::MultilinearForm;
    verbosity::Int = 0,
    factor = 1)

    FEBarrays = Array{AbstractVector,1}(undef, length(FEB))
    offsets = zeros(Int,length(FEB))
    for j = 1 : length(FEB)
        FEBarrays[j] = FEB[j].entries
        offsets[j] = FEB[j].offset
    end

    assemble!(b.entries, FEBarrays, MLF; verbosity = verbosity, factor = factor, offset = b.offset, offsets2 = offsets)
end




"""
````
assemble!(
    A::AbstractArray{T,2},
    NLF::NonlinearForm{T, AT},
    FEB::Array{<:FEVectorBlock,1};         # coefficients for each operator
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false
````

Assembly of a NonlinearForm NLF into given two-dimensional AbstractArray (e.g. FEMatrixBlock).
"""
function assemble!(
    A::AbstractArray{T,2},
    NLF::NonlinearForm{T, AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    offsetX = 0,
    offsetY = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # extract finite element spaces and operators
    FE = NLF.FE1
    push!(FE, NLF.FE2)
    nFE::Int = length(FE)
    operators = NLF.operator1
    push!(operators, NLF.operator2)
    
    # get adjacencies
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    for j = 1 : nFE
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, operators[j]))
    end
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = NLF.action
    regions = parse_regions(NLF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(NLF, operators, FE, regions, nFE, action.bonus_quadorder, verbosity - 1)
 
 
    # get size informations
    ncomponents = zeros(Int,length(FE))
    offsets = zeros(Int,length(FE)+1)
    maxdofs = 0
    for j = 1 : nFE
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[1,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]
    maxdofs2 = max_num_targets_per_source(xItemDofs[end])

 
    maxnweights = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end-1]) # heap for action input
    end
    action_input2 = zeros(T,offsets[end-1])

    # loop over items
    EG4item::Int = 0
    EG4dofitem1::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    EG4dofitem2::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    dofitems1::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    dofitems2::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem1::Array{Int,1} = [1,1] # local item position in dofitem
    itempos4dofitem2::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem1::Array{Int,1} = [1,2] # local orientation
    orientation4dofitem2::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem1::Array{Int,1} = [0,0] # coefficients for operator
    coefficient4dofitem2::Array{Int,1} = [0,0] # coefficients for operator
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitem::Int = 0
    coeffs::Array{T,1} = zeros(T,maxdofs)
    dofs::Array{Int,1} = zeros(Int,maxdofs)
    dofs2::Array{Int,1} = zeros(Int,maxdofs2)
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1,1,1,1]
    basisevaler4dofitem1 = Array{Any,1}(undef,nFE)
    for j = 1 : nFE
        basisevaler4dofitem1[j] = basisevaler[1,j,1,1]
    end
    basisevaler4dofitem2 = basisevaler[1,end,1,1]
    basisvals::Array{T,3} = basisevaler4dofitem.cvals
    temp::T = 0 # some temporary variable
    localmatrix::Array{T,2} = zeros(T,maxdofs,maxdofs2)
    acol::Int = 0
    arow::Int = 0


    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherweise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations for ansatz function
        EG4item = dii4op[1](dofitems1, EG4dofitem1, itempos4dofitem1, coefficient4dofitem1, orientation4dofitem1, item)

        # get quadrature weights for integration domain
        weights = qf[EG4dofitem1[1]].w

        # fill action input with evluation of current solution
        # given by coefficient vectors
        for FEid = 1 : nFE - 1

            # get information on dofitems
            weights = qf[EG4item].w
            for di = 1 : length(dofitems1)
                dofitem = dofitems1[di]
                if dofitem != 0
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem1[di],FEid,itempos4dofitem1[di],orientation4dofitem1[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem1[di]]
                    for j=1:ndofs4dofitem
                        fdof = xItemDofs[FEid][j,dofitem]
                        coeffs[j] = FEB[FEid][fdof]
                    end

                    for i in eachindex(weights)
                        if FEid == 1 && di == 1
                            fill!(action_input[i], 0)
                        end
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i; offset = offsets[FEid], factor = coefficient4dofitem1[di])
                    end  
                end
            end
        end

        # at this point the action_input at each quadrature point contains information on the last solution
        # also no jump operators for the test function are allowed currently

        # get dof information for test function
        dii4op[end](dofitems2, EG4dofitem2, itempos4dofitem2, coefficient4dofitem2, orientation4dofitem2, item)
        di = 1
        dj = 1
        dofitem1 = dofitems1[di]
        dofitem2 = dofitems2[dj]
        ndofs4item1 = ndofs4EG[1][EG4dofitem1[di]]
        ndofs4item2 = ndofs4EG[2][EG4dofitem2[dj]]

        # update FEbasisevalers for ansatz function
        for FEid = 1 : nFE - 1
            basisevaler4dofitem1[FEid] = basisevaler[EG4dofitem1[di],FEid,itempos4dofitem1[di],orientation4dofitem1[di]]
            update!(basisevaler4dofitem1[FEid],dofitem1)
        end

        # update FEbasisevalers for test function
        basisevaler4dofitem2 = basisevaler[EG4dofitem2[dj],end,itempos4dofitem2[dj],orientation4dofitem2[dj]]
        basisvals = basisevaler4dofitem2.cvals
        update!(basisevaler4dofitem2,dofitem2)

        # update action on dofitem
        update!(action, basisevaler4dofitem1[1], dofitem1, item, regions[r])

        # update dofs
        for j=1:ndofs4item1
            dofs[j] = xItemDofs[1][j,dofitem1]
        end
        for j=1:ndofs4item2
            dofs2[j] = xItemDofs[end][j,dofitem2]
        end

        for i in eachindex(weights)
            for dof_i = 1 : ndofs4item1

                for FEid = 1 : nFE - 1
                    eval!(action_input2, basisevaler4dofitem1[FEid], dof_i, i; offset = offsets[FEid])
                end

                apply_action!(action_result, action_input[i], action_input2, action, i)
                action_result .*= weights[i]

                for dof_j = 1 : ndofs4item2
                    temp = 0
                    for k = 1 : action_resultdim
                        temp += action_result[k] * basisvals[k,dof_j,i]
                    end
                    temp *= coefficient4dofitem2[dj]
                    localmatrix[dof_i,dof_j] += temp
                end
            end 
        end

        localmatrix .*= xItemVolumes[item] * factor

        # copy localmatrix into global matrix
        for dof_i = 1 : ndofs4item1
            arow = dofs[dof_i] + offsetX
            for dof_j = 1 : ndofs4item2
                if localmatrix[dof_i,dof_j] != 0
                    acol = dofs2[dof_j] + offsetY
                    if transposed_assembly == true
                        _addnz(A,acol,arow,localmatrix[dof_i,dof_j],1)
                    else 
                        _addnz(A,arow,acol,localmatrix[dof_i,dof_j],1)  
                    end
                end
            end
        end
        
        fill!(localmatrix,0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

    return nothing
end


## wrapper for FEMatrixBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    A::FEMatrixBlock,
    NLF::NonlinearForm,
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false)

    assemble!(A.entries, NLF, FEB; verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, offsetX = A.offsetX, offsetY = A.offsetY)
end




"""
````
assemble!(
    b::AbstractVector,
    NLF::NonlinearForm{T, AT},
    FEB::Array{<:FEVectorBlock,1};         # coefficients for each operator
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false
````

Assembly of a NonlinearForm NLF into given AbstractVector (e.g. FEMatrixBlock).
"""
function assemble!(
    b::AbstractVector,
    NLF::NonlinearForm{T, AT},
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false,
    offset = 0) where {T<: Real, AT <: AbstractAssemblyType}

    # extract finite element spaces and operators
    FE = NLF.FE1
    push!(FE, NLF.FE2)
    nFE::Int = length(FE)
    operators = NLF.operator1
    push!(operators, NLF.operator2)
    
    # get adjacencies
    xItemVolumes::Array{T,1} = FE[1].xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    for j = 1 : nFE
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], DofitemAT4Operator(AT, operators[j]))
    end
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems = length(xItemVolumes)

    # prepare assembly
    action = NLF.action
    regions = parse_regions(NLF.regions, xItemRegions)
    EG, ndofs4EG, qf, basisevaler, dii4op = prepare_assembly(NLF, operators, FE, regions, nFE, action.bonus_quadorder, verbosity - 1)
 
 
    # get size informations
    ncomponents = zeros(Int,length(FE))
    offsets = zeros(Int,length(FE)+1)
    maxdofs = 0
    for j = 1 : nFE
        ncomponents[j] = get_ncomponents(eltype(FE[j]))
        maxdofs = max(maxdofs, max_num_targets_per_source(xItemDofs[j]))
        offsets[j+1] = offsets[j] + size(basisevaler[1,j,1,1].cvals,1)
    end
    action_resultdim::Int = action.argsizes[1]
    maxdofs2 = max_num_targets_per_source(xItemDofs[end])

 
    maxnweights = 0
    for j = 1 : length(qf)
        maxnweights = max(maxnweights, length(qf[j].w))
    end
    action_input = Array{Array{T,1},1}(undef,maxnweights)
    for j = 1 : maxnweights
        action_input[j] = zeros(T,offsets[end-1]) # heap for action input
    end

    # loop over items
    EG4item::Int = 0
    EG4dofitem::Array{Int,1} = [1,1] # EG id of the current item with respect to operator
    dofitems::Array{Int,1} = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    coefficient4dofitem::Array{T,1} = [0,0] # coefficients for operator
    orientation4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    ndofs4dofitem::Int = 0 # number of dofs for item
    dofitem::Int = 0
    coeffs::Array{T,1} = zeros(T,maxdofs)
    dofs::Array{Int,1} = zeros(Int,maxdofs)
    action_result::Array{T,1} = zeros(T,action_resultdim) # heap for action output
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem = basisevaler[1,1,1,1]
    basisevaler4dofitem1 = Array{Any,1}(undef,nFE)
    for j = 1 : nFE
        basisevaler4dofitem1[j] = basisevaler[1,j,1,1]
    end
    basisvals::Array{T,3} = basisevaler4dofitem.cvals
    temp::T = 0 # some temporary variable
    localb = zeros(T,maxdofs)
    bdof::Int = 0


    # note: at the moment we expect that all FE[1:end-1] are the same !
    # otherweise more than one MatrixBlock has to be assembled and we need more offset information
    # hence, this only can handle nonlinearities at the moment that depend on one unknown of the PDEsystem

    for item = 1 : nitems
    for r = 1 : length(regions)
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # get dofitem informations for ansatz function
        EG4item = dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

        # get quadrature weights for integration domain
        weights = qf[EG4dofitem[1]].w

        # fill action input with evluation of current solution
        # given by coefficient vectors
        for FEid = 1 : nFE - 1

            # get information on dofitems
            weights = qf[EG4item].w
            for di = 1 : length(dofitems)
                dofitem = dofitems[di]
                if dofitem != 0
                    # update FEbasisevaler on dofitem
                    basisevaler4dofitem = basisevaler[EG4dofitem[di],FEid,itempos4dofitem[di],orientation4dofitem[di]]
                    update!(basisevaler4dofitem, dofitem)

                    # update coeffs on dofitem
                    ndofs4dofitem = ndofs4EG[FEid][EG4dofitem[di]]
                    for j=1:ndofs4dofitem
                        bdof = xItemDofs[FEid][j,dofitem]
                        coeffs[j] = FEB[FEid][bdof]
                    end

                    for i in eachindex(weights)
                        if FEid == 1 && di == 1
                            fill!(action_input[i], 0)
                        end
                        eval!(action_input[i], basisevaler4dofitem, coeffs, i; offset = offsets[FEid], factor = coefficient4dofitem[di])
                    end  
                end
            end
        end

        # at this point the action_input at each quadrature point contains information on the last solution
        # also no jump operators for the test function are allowed currently

        di = 1
        dofitem = dofitems[di]
        ndofs4item = ndofs4EG[1][EG4dofitem[di]]

        # update FEbasisevalers for test function
        basisevaler4dofitem = basisevaler[EG4dofitem[di],end,itempos4dofitem[di],di]
        basisvals = basisevaler4dofitem.cvals
        update!(basisevaler4dofitem,dofitem)

        # update action on dofitem
        update!(action, basisevaler4dofitem, dofitem, item, regions[r])

        # update dofs
        for j=1:ndofs4item
            dofs[j] = xItemDofs[1][j,dofitem]
        end

        for i in eachindex(weights)
            apply_action!(action_result, action_input[i], action, i)
            action_result .*= weights[i]

            for dof_j = 1 : ndofs4item
                temp = 0
                for k = 1 : action_resultdim
                    temp += action_result[k] * basisvals[k,dof_j,i]
                end
                temp *= coefficient4dofitem[di]
                localb[dof_j] += temp
            end
        end

        localb .*= xItemVolumes[item] * factor

        # copy localmatrix into global matrix
        for dof_i = 1 : ndofs4item
            bdof = dofs[dof_i] + offset
            b[bdof] += localb[dof_i]          
        end
        
        fill!(localb,0.0)
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

    return nothing
end


## wrapper for FEVectorBlock to avoid use of setindex! functions of FEMAtrixBlock
function assemble!(
    b::FEVectorBlock,
    NLF::NonlinearForm,
    FEB::Array{<:FEVectorBlock,1};
    verbosity::Int = 0,
    factor = 1,
    transposed_assembly::Bool = false)

    assemble!(b.entries, NLF, FEB; verbosity = verbosity, factor = factor, transposed_assembly = transposed_assembly, offset = b.offset)
end





"""
````
function L2ErrorIntegrator(
    T::Type{<:Real},
    compare_data::UserData{AbstractDataFunction},
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    time = 0)
````

Creates an ItemIntegrator that compares FEVectorBlock operator-evaluations against the given compare_data and returns the L2-error.
"""
function L2ErrorIntegrator(T::Type{<:Real},
    compare_data::UserData{AbstractDataFunction},
    operator::Type{<:AbstractFunctionOperator};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    time = 0)

    ncomponents::Int = compare_data.dimensions[1]
    temp = zeros(T,ncomponents)
    function L2error_function(result,input,x)
        eval!(temp,compare_data,x,time)
        result[1] = 0
        for j=1:ncomponents
            result[1] += (temp[j] - input[j])^2
        end    
    end    
    action_kernel = ActionKernel(L2error_function, [1,compare_data.dimensions[2]]; name = "L2 error kernel", dependencies = "X", quadorder = 2 * compare_data.quadorder)
    return ItemIntegrator{T,AT}([operator], Action(T, action_kernel), [0])
end