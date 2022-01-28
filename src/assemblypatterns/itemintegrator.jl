
"""
$(TYPEDEF)

itemintegrator assembly pattern type
"""
abstract type APT_ItemIntegrator <: AssemblyPatternType end

function Base.show(io::IO, ::Type{APT_ItemIntegrator})
    print(io, "ItemIntegrator")
end

"""
$(TYPEDSIGNATURES)

Creates an ItemIntegrator assembly pattern based on:

- operators : operators that should be evaluated for the coressponding FESpace (last one refers to test function)
- action    : an Action with kernel of interface (result, input, kwargs) that takes input (= all but last operator evaluations) and computes result to be dot-producted with test function evaluation
              (if no action is specified, the full input vector is dot-producted with the test function operator evaluation)

Optional arguments:
- T         : expected NumberType for evaluation output
- AT        : specifies on which entities of the grid the ItemINtegrator is evaluated
- regions   : specifies in which regions the operator should assemble, default [0] means all regions
- name      : name for this LinearForm that is used in print messages

"""
function ItemIntegrator(operators, action = NoAction(); T = Float64, AT = ON_CELLS, regions = [0], name = "ItemIntegrator")
    return AssemblyPattern{APT_ItemIntegrator, T, AT}(name,Array{FESpace{Float64,Int32},1}([]),operators,action,1:length(operators),regions)
end

"""
$(TYPEDSIGNATURES)

Creates an ItemIntegrator that compares discrete FEVectorBlock operator-evaluations against the given compare_data and returns the L2-error
|| compare_data(x) - factor*discrete(x) ||.
If quadorder is left on "auto" two times the quadorder of the data is used in the evaluation.
"""
function L2ErrorIntegrator(
    compare_data::UserData{AbstractDataFunction},
    operator = Identity;
    T = Float64,
    quadorder = "auto",
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    factor = 1,
    regions = [0],
    time = 0)

    ninputs::Int = 1
    ops = operator
    if typeof(operator) <: DataType
        ops = [operator]
    else
        ninputs = length(operator)
    end

    ncomponents::Int = compare_data.dimensions[1]
    temp::Array{T,1} = zeros(T,ncomponents)
    
    function L2error_function(result,input,x)
        fill!(temp,0)
        eval_data!(temp,compare_data,x,time)
        result[1] = 0
        for j=1:ncomponents
            for i = 1 : ninputs
                temp[j] -= input[(i-1)*ncomponents+j]*factor
            end
            result[1] += temp[j].^2
        end    
        return nothing
    end    
    if quadorder == "auto"
        quadorder = 2 * compare_data.quadorder
    end
    if name == "auto"
        name = "L2 error ($(compare_data.name))"
    end
    action = Action(L2error_function, [1,ninputs*compare_data.dimensions[1]]; Tv = T, name = name, dependencies = "X", bonus_quadorder = quadorder)
    return ItemIntegrator(ops, action; T = T, AT = AT, regions = regions, name = name)
end

"""
$(TYPEDSIGNATURES)

Creates an ItemIntegrator that computes the L2 norm of an operator evaluation where ncomponents is the expected length of the operator evaluation.
"""
function L2NormIntegrator(
    ncomponents::Int,
    operator;
    T = Float64,
    AT::Type{<:AssemblyType} = ON_CELLS,
    name = "L2 norm",
    quadorder = 2,
    regions = [0])

    ninputs::Int = 1
    ops = operator
    if typeof(operator) <: DataType
        ops = [operator]
    else
        ninputs = length(operator)
    end

    temp::Array{T,1} = zeros(T,ncomponents)
    function L2norm_function(result,input)
        fill!(temp,0)
        for j=1:ncomponents, i = 1 : ninputs
            temp[j] += input[(i-1)*ncomponents+j]
        end
        result[1] = 0
        for j=1:ncomponents
            result[1] += temp[j]^2
        end    
    end    
    action = Action(L2norm_function, [1,ninputs*ncomponents]; name = "L2 norm kernel", dependencies = "", bonus_quadorder = quadorder)
    return ItemIntegrator(ops, action; T = T, AT = AT, regions = regions, name = name)
end

"""
$(TYPEDSIGNATURES)

Creates an ItemIntegrator that computes the L2 norm difference between two arguments evalauted with the same operator (or with different operators if operator is an array) where ncomponents is the expected length of each operator evaluation.
Note that all arguments in an evaluation call need to be defined on the same grid !
"""
function L2DifferenceIntegrator(
    ncomponents::Int,
    operator;
    AT::Type{<:AssemblyType} = ON_CELLS,
    T = Float64,
    name = "L2 difference",
    quadorder = 2,
    regions = [0])
    function L2difference_function(result,input)
        result[1] = 0
        for j=1:ncomponents
            result[1] += (input[j]-input[ncomponents+j])^2
        end    
    end    
    action = Action(L2difference_function, [1,2*ncomponents]; name = "L2 difference kernel", dependencies = "", bonus_quadorder = quadorder)
    if typeof(operator) <: DataType
        return ItemIntegrator([operator, operator], action; T = T, AT = AT, regions = regions, name = name)
    else
        return ItemIntegrator([operator[1], operator[2]], action; T = T, AT = AT, regions = regions, name = name)
    end
end



"""
````
function evaluate!(
    b::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Union{<:FEVector{T,Tv,Ti},<:FEVectorBlock{T,Tv,Ti},Array{<:FEVectorBlock{T,Tv,Ti},1}};
    skip_preps::Bool = false) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AssemblyType, Tv, Ti}
````

Evaluation of an ItemIntegrator assembly pattern with given FEVectorBlock or FEVector FEB into given two-dimensional Array b.
"""
function evaluate!(
    b::AbstractArray{T,2},
    AP::AssemblyPattern{APT,T,AT},
    FEB::Union{<:FEVector{T,Tv,Ti},<:FEVectorBlock{T,Tv,Ti},Array{<:FEVectorBlock{T,Tv,Ti},1}};
    skip_preps::Bool = false) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AssemblyType, Tv, Ti}

    if typeof(FEB) <: FEVectorBlock
        FEB = [FEB]
    end

    # prepare assembly
    nFE::Int = length(FEB)
    if !skip_preps
        FE = Array{FESpace{Tv,Ti},1}(undef, nFE)
        for j = 1 : nFE
            FE[j] = FEB[j].FES
        end
        @assert length(FEB) == length(AP.operators)
        prepare_assembly!(AP, FE)
    end
    AM::AssemblyManager{T} = AP.AM
    xItemVolumes::Array{Tv,1} = FEB[1].FES.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemRegions::GridRegionTypes{Ti} = FEB[1].FES.xgrid[GridComponentRegions4AssemblyType(AT)]
    nitems::Int = length(xItemVolumes)

    # prepare action
    action = AP.action
    maxnweights::Int = get_maxnqweights(AM)
    if typeof(action) <: NoAction
        action_resultdim = size(get_basisevaler(AM, 1, 1).cvals,1)
        action_input = Array{Array{T,1},1}(undef,maxnweights)
        if length(FEB) > 1
            @warn "NoAction() used in ItemIntegrator with more than one argument!"
        end
        for j = 1 : maxnweights
            action_input[j] = zeros(T,action_resultdim) # heap for action input
        end
        action_result = zeros(T,action_resultdim)
    else
        action_resultdim::Int = action.argsizes[1]
        action_input = Array{Array{T,1},1}(undef,maxnweights)
        for j = 1 : maxnweights
            action_input[j] = zeros(T,action.argsizes[2]) # heap for action input
        end
        action_result = action.val
    end

    if AP.regions != [0]
        @logmsg MoreInfo "Evaluating $(AP.name) for $((p->p.name).(FEB)) ($AT in regions = $(AP.regions))"
    else
        @logmsg MoreInfo "Evaluating $(AP.name) for $((p->p.name).(FEB)) ($AT)"
    end
    @debug AP

    # loop over items
    offsets::Array{Int,1} = zeros(Int,nFE+1)
    maxdofs::Array{Int,1} = get_maxndofs(AM)
    basisevaler::Array{FEBasisEvaluator,1} = Array{FEBasisEvaluator,1}(undef,nFE)
    for j = 1 : nFE
        basisevaler[j] = get_basisevaler(AM, j, 1)
        offsets[j+1] = offsets[j] + get_basisdim(AM, j)
    end

    if length(action_input[1]) != offsets[end]
        @warn "operator input length $(length(action_input[1])) does not match calculated offsets ($(offsets[end]))"
    end
    basisxref::Array{Array{T,1},1} = basisevaler[1].xref
    maxdofitems::Array{Int,1} = get_maxdofitems(AM)
    coeffs::Array{T,1} = zeros(T,sum(maxdofs[1:end]))
    weights::Array{T,1} = get_qweights(AM)
    regions::Array{Int,1} = AP.regions
    allitems::Bool = (regions == [0])
    nregions::Int = length(regions)
    for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if allitems || xItemRegions[item] == regions[r]

        # update assembly manager (also updates necessary basisevaler)
        update_assembly!(AM, item)
        weights = get_qweights(AM)

        # assemble all operators into action_input
        for FEid = 1 : nFE
            for di = 1 : maxdofitems[FEid]
                if AM.dofitems[FEid][di] != 0
                    # get correct basis evaluator for dofitem (was already updated by AM)
                    basisevaler[FEid] = get_basisevaler(AM, FEid, di)

                    # get coefficients of FE number FEid on current dofitem
                    get_coeffs!(coeffs, FEB[FEid], AM, FEid, di)
                    coeffs .*= AM.coeff4dofitem[FEid][di]

                    # write evaluation of operator of current FE into action_input
                    for i in eachindex(weights)
                        eval_febe!(action_input[i], basisevaler[FEid], coeffs, i, offsets[FEid])
                    end  
                end
            end
        end

        if typeof(action) <: NoAction
            for i in eachindex(weights)
                for j = 1 : action_resultdim
                    b[j,item] += action_input[i][j] * weights[i] * xItemVolumes[item]
                end
                fill!(action_input[i], 0)
            end  
        else
            # update action on item/dofitem (of first operator)
            if is_itemdependent(action)
                action.item[1] = item
                action.item[2] = AM.dofitems[1][1]
                action.item[3] = xItemRegions[item]
            end
            basisxref = basisevaler[1].xref
            # apply action to FEVector and accumulate
            for i in eachindex(weights)
                if is_xdependent(action)
                    update_trafo!(basisevaler[1].L2G, item)
                    eval_trafo!(action.x, basisevaler[1].L2G, basisevaler[1].xref[i])
                end
                if is_xrefdependent(action)
                    action.xref = basisevaler[1].xref[i]
                end
                #apply_action!(action_result, action_input[i], action, i, basisxref[i])
                eval_action!(action, action_input[i])
                for j = 1 : action_resultdim
                    b[j,item] += action_result[j] * weights[i] * xItemVolumes[item]
                end
                fill!(action_input[i], 0)
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
    AP::AssemblyPattern{APT,T,AT},
    FEB::Union{<:FEVector{T,Tv,Ti},<:FEVectorBlock{T,Tv,Ti},Array{<:FEVectorBlock{T,Tv,Ti},1}};
    skip_preps::Bool = false) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AssemblyType, Tv, Ti}

````

Evaluation of an ItemIntegrator assembly pattern with given FEVectorBlock or FEVector FEB, only returns accumulation over all items.
"""
function evaluate(
    AP::AssemblyPattern{APT,T,AT},
    FEB::Union{<:FEVector{T,Tv,Ti},<:FEVectorBlock{T,Tv,Ti},Array{<:FEVectorBlock{T,Tv,Ti},1}};
    skip_preps::Bool = false) where {APT <: APT_ItemIntegrator, T<: Real, AT <: AssemblyType, Tv, Ti}

    if typeof(FEB) <: FEVectorBlock
        FE = [FEB.FES]
    else
        nFE = length(FEB)
        if !skip_preps
            FE = Array{FESpace{Tv,Ti},1}(undef, nFE)
            for j = 1 : nFE
                FE[j] = FEB[j].FES
            end
        end
    end

    # prepare assembly
    @assert length(FE) == length(AP.operators)
    prepare_assembly!(AP,FE)
    AM::AssemblyManager{T} = AP.AM

    # quick and dirty : we mask the resulting array as an AbstractArray{T,2} using AccumulatingVector
    # and use the itemwise evaluation above
    if typeof(AP.action) <: NoAction
        resultdim = size(get_basisevaler(AM, 1, 1).cvals,1)
    else
        resultdim = AP.action.argsizes[1]
    end
    AV = AccumulatingVector{T}(zeros(T,resultdim), 0)

    evaluate!(AV, AP, FEB; skip_preps = true)
    
    if resultdim == 1
        return AV.entries[1]
    else
        return AV.entries
    end
end