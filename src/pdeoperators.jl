
# type to steer when a PDE block is (re)assembled
abstract type AbstractAssemblyTrigger end
abstract type AssemblyAuto <: AbstractAssemblyTrigger end           # triggers automatic decision when block is reassembled at solver configuration level
abstract type AssemblyFinal <: AbstractAssemblyTrigger end          # is only assembled after solving
abstract type AssemblyAlways <: AbstractAssemblyTrigger end         # is always (re)assembled
    abstract type AssemblyEachTimeStep <: AssemblyAlways end        # is (re)assembled in each timestep
        abstract type AssemblyInitial <: AssemblyEachTimeStep end   # is only assembled in initial assembly
            abstract type AssemblyNever <: AssemblyInitial end      # is never assembled



#######################
# AbstractPDEOperator #
#######################
#
# to describe operators in the (weak form of the) PDE
#
# some intermediate layer that knows nothing of the FE discretisations
# but know about their connectivities to help devise fixpoint iterations
# and triggers certain AssemblyPatterns when called for assembly!
#
# USER-DEFINED ABSTRACTPDEOPERATORS
# might be included if they implement the following interfaces
#
#   (1) to specify what is assembled into the corressponding MatrixBlock:
#       assemble!(A::FEMatrixBlock, SC::SolverConfig, j::Int, k::Int, o::Int, O::AbstractPDEOperator, CurrentSolution::FEVector; time, factor)
#       assemble!(b::FEVectorBlock, SC::SolverConfig, j::Int, k::Int, o::Int, O::AbstractPDEOperator, CurrentSolution::FEVector; time, factor)
#       assemble!(b::FEVectorBlock, SC::SolverConfig, j::Int, o::Int, O::AbstractPDEOperator, CurrentSolution::FEVector; time, factor)
#
#   (2) to allow SolverConfig to check if operator is nonlinear, timedependent:
#       Bool, Bool, AssemblyTrigger = check_PDEoperator(O::AbstractPDEOperator)
#
#   (3) to allow SolverConfig to check if operator depends on unknown number arg:
#       Bool = check_dependency(O::AbstractPDEOperator, arg::Int)
# 


abstract type AbstractPDEOperator end
abstract type NoConnection <: AbstractPDEOperator end # => empy block in matrix

has_copy_block(::AbstractPDEOperator) = false
has_storage(::AbstractPDEOperator) = false

# check if operator causes nonlinearity or time-dependence
function check_PDEoperator(O::AbstractPDEOperator, involved_equations)
    return false, false, AssemblyInitial
end

# check if operator also depends on arg (additional to the argument relative to position in PDEDescription)
function check_dependency(O::AbstractPDEOperator, arg::Int)
    return false
end


"""
$(TYPEDEF)

common structures for all finite element operators that are assembled with GradientRobustMultiPhysics; better look at the AssemblyPatternType and the constructors
"""
mutable struct PDEOperator{T <: Real, APT <: AssemblyPatternType, AT <: AssemblyType} <: AbstractPDEOperator
    name::String
    operators4arguments::Array{DataType,1}
    active_recast::Bool
    action::Union{AbstractAction, AbstractNonlinearFormHandler}
    action_rhs::Union{AbstractAction, AbstractNonlinearFormHandler}
    action_eval::Union{AbstractAction, AbstractNonlinearFormHandler}
    apply_action_to::Array{Int,1}
    fixed_arguments::Array{Int,1}
    fixed_arguments_ids::Array{Int,1}
    newton_arguments::Array{Int,1}
    factor::T
    transpose_factor::T
    regions::Array{Int,1}
    transposed_assembly::Bool
    transposed_copy::Bool
    store_operator::Bool
    assembly_trigger::Type{<:AbstractAssemblyTrigger}
    storage_init::Bool
    storage_A::Union{Nothing,AbstractMatrix{T},FEMatrix{T}}
    storage_b::Union{Nothing,AbstractVector{T},FEVector{T}}
    PDEOperator{T,APT,AT}(name,ops) where {T <: Real, APT <: AssemblyPatternType, AT <: AssemblyType} = new{T,APT,AT}(name,ops,false,NoAction(),NoAction(),NoAction(),[1],[],[],[],1,1,[0],false,false,false,AssemblyAuto,false)
    PDEOperator{T,APT,AT}(name,ops,factor) where {T,APT,AT} = new{T,APT,AT}(name,ops,false,NoAction(),NoAction(),NoAction(),[1],[],[],[],factor,factor,[0],false,false,false,AssemblyAuto,false)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor) where {T,APT,AT} = new{T,APT,AT}(name,ops,false,action,action,action,apply_to,[],[],[],factor,factor,[0],false,false,false,AssemblyAuto,false)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions) where {T,APT,AT} = new{T,APT,AT}(name,ops,false,action,action,action,apply_to,[],[],[],factor,factor,regions,false,false,false,AssemblyAuto,false)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions,store) where {T,APT,AT} = new{T,APT,AT}(name,ops,false,action,action,action,apply_to,[],[],[],factor,factor,regions,false,false,store,AssemblyAuto,false)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions,store,assembly_trigger) where {T,APT,AT} = new{T,APT,AT}(name,ops,false,action,action,action,apply_to,[],[],[],factor,factor,regions,false,false,store,assembly_trigger,false)
end 

get_pattern(::AbstractPDEOperator) = APT_Undefined
get_pattern(::PDEOperator{T,APT,AT}) where{T,APT,AT} = APT

## provisoric copy method for PDEoperators (needed during assignment of nonlinear operators, where each copy is related to partial derivatives of each related unknown)
export copy
function Base.copy(O::PDEOperator{T,APT,AT}) where{T,APT,AT}
    cO = PDEOperator{T,APT,AT}(O.name,copy(O.operators4arguments))
    cO.action = O.action # deepcopy would lead to errors !!!
    cO.action_rhs = O.action_rhs # deepcopy would lead to errors !!!
    cO.action_eval = O.action_eval # deepcopy would lead to errors !!! 
    cO.active_recast = O.active_recast
    cO.apply_action_to = copy(O.apply_action_to)
    cO.fixed_arguments = copy(O.fixed_arguments)
    cO.fixed_arguments_ids = copy(O.fixed_arguments_ids)
    cO.newton_arguments = copy(O.newton_arguments)
    cO.factor = O.factor
    cO.regions = copy(O.regions)
    cO.transposed_assembly = O.transposed_assembly
    cO.store_operator = O.store_operator
    cO.assembly_trigger = O.assembly_trigger
    return cO
end

function Base.show(io::IO, O::PDEOperator)
    println(io,"\n\toperator name = $(O.name)")
	println(io,"\toperator type = $(typeof(O).parameters[2]) (T = $(typeof(O).parameters[1]))")
    println(io,"\toperator span = $(typeof(O).parameters[3]) (regions = $(O.regions))")
    println(io,"\toperator oprs = $(O.operators4arguments)")
    if !(typeof(O.action) <: NoAction)
        println(io,"\toperator actn = $(O.action.name) (apply_to = $(O.apply_action_to) size = $(O.action.argsizes))")
    end
    println(io,"\ttranspose A = $(O.transposed_assembly)")
    println(io,"\ttranspose C = $(O.transposed_copy)")
    println(io,"\tfixed ar/id = $(O.fixed_arguments) / $(O.fixed_arguments_ids)")
end

has_copy_block(O::PDEOperator) = O.transposed_copy
has_storage(O::PDEOperator) = O.store_operator
function check_PDEoperator(O::PDEOperator, involved_equations)
    nonlinear = length(O.fixed_arguments_ids) > 0
    timedependent = typeof(O.action) <: NoAction ? false : is_timedependent(O.action)
    if O.assembly_trigger <: AssemblyAuto
        assembly_trigger = AssemblyInitial
        if timedependent
            assembly_trigger = AssemblyEachTimeStep
        end
        if nonlinear
            assembly_trigger = AssemblyAlways
        end
    else 
        assembly_trigger = O.assembly_trigger
    end
    return nonlinear, timedependent, assembly_trigger
end
# check if operator also depends on arg (additional to the argument(s) relative to position in PDEDescription)
function check_dependency(O::PDEOperator, arg::Int)
    return arg in O.fixed_arguments_ids
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = κ (∇u,∇v) where kappa is some constant (diffusion) coefficient.
"""
function LaplaceOperator(κ = 1.0; name = "auto", AT::Type{<:AssemblyType} = ON_CELLS, ∇ = Gradient, regions::Array{Int,1} = [0], store::Bool = false)
    if name == "auto"
        name = "(∇#A,∇#T)"
        if typeof(κ) <: Real
            if κ != 1
                name = "$κ " * name
            end
        end
    end
    if typeof(κ) <: Real
        O = PDEOperator{Float64, APT_SymmetricBilinearForm, AT}(name,[∇, ∇], NoAction(), [1], κ, regions, store, AssemblyInitial)
        return O
    else
        @error "No standard Laplace operator definition for this type of κ available, please define your own action and PDEOperator with it."
    end
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform a(u,v) = (αu,v) or (u,αv) with some coefficient α that can be a number or an AbstractDataFunction.
"""
function ReactionOperator(α = 1.0, ncomponents = 1; name = "auto", AT::Type{<:AssemblyType} = ON_CELLS, id = Identity, regions::Array{Int,1} = [0], store::Bool = false)
    if name == "auto"
        name = "(#A,#T)"
        if typeof(α) <: Real
            if α != 1.0
                name = "$α " * name
            end
        elseif typeof(α) <: AbstractUserDataType
            name = "(α #A,#T)"
        end
    end
    if typeof(α) <: Real
        return PDEOperator{Float64, APT_SymmetricBilinearForm, AT}(name,[id,id], NoAction(), [1], α, regions, store, AssemblyInitial)
    elseif typeof(α) <: AbstractUserDataType
        function reaction_kernel(result, input, kwargs...)
            # evaluate alpha
            eval_data!(α, kwargs...)
            # compute alpha*u
            for j = 1 : ncomponents
                result[j] = α.val[j] * input[j]
            end
        end    
        action = Action(reaction_kernel, [ncomponents, ncomponents]; dependencies = dependencies(α), bonus_quadorder = α.bonus_quadorder)
        return PDEOperator{Float64, APT_BilinearForm, AT}(name, [id, id], action, [1], 1, regions, store, AssemblyAuto)
    else
        @error "No standard reaction operator definition for this type of α available, please define your own action and PDEOperator with it."
    end
end


"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (A(operator(u)), id(v)) and assembles a second transposed block at the block of the transposed PDE coordinates. It is intended to use
to render one unknown of the PDE the Lagrange multiplier for another unknown by putting this operator on the coressponding subdiagonal block of the PDE description.

Example: LagrangeMultiplier(Divergence) is used to render the pressure the LagrangeMultiplier for the velocity divergence constraint in the Stokes prototype.
"""
function LagrangeMultiplier(operator::Type{<:AbstractFunctionOperator}; name = "auto", AT::Type{<:AssemblyType} = ON_CELLS, action::AbstractAction = NoAction(), regions::Array{Int,1} = [0], store::Bool = false, factor = -1)
    if name == "auto"
        name = "(#A, $operator(#T))"
        if factor == -1
            name = "-" * name
        elseif factor != 1
            name = "$factor " * name
        end
    end
    O = PDEOperator{Float64, APT_BilinearForm, AT}(name,[operator, Identity], action, [1], factor, regions, store, AssemblyInitial)
    O.transposed_copy = true
    return O
end


"""
$(TYPEDSIGNATURES)

constructor for a bilinearform a(u,v) = (μ ∇u,∇v) where C is the 1D stiffness tensor for given μ.
    
"""
function HookStiffnessOperator1D(μ; name = "(μ ∇#A,∇#T)", regions::Array{Int,1} = [0], ∇ = TangentialGradient, store::Bool = false)
    return PDEOperator{Float64, APT_BilinearForm, ON_CELLS}(name,[∇, ∇], NoAction(), [1], μ, regions, store, AssemblyInitial)
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform a(u,v) = (C ϵ(u), ϵ(v)) where C is the 2D stiffness tensor
for isotropic media in Voigt notation, i.e.
ℂ ϵ(u) = 2 μ ϵ(u) + λ tr(ϵ(u)) for Lame parameters μ and λ
    
    In Voigt notation ℂ is a 3 x 3 matrix
    ℂ = [c11,c12,  0
         c12,c11,  0
           0,  0,c33]
    
    where c33 = μ, c12 = λ and c11 = 2*c33 + c12

Note: ϵ is the symmetric part of the gradient (in Voigt notation)
    
"""
function HookStiffnessOperator2D(μ, λ; 
    name = "(ℂ(μ,λ) ϵ(#A),ϵ(#T))", 
    AT::Type{<:AssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0], 
    ϵ = SymmetricGradient{1}, 
    store::Bool = false)

    @assert ϵ <: SymmetricGradient

    function tensor_apply_2d(result, input)
        result[1] = (λ + 2*μ)*input[1] + λ*input[2]
        result[2] = (λ + 2*μ)*input[2] + λ*input[1]
        result[3] = μ*input[3]
        return nothing
    end   
    action = Action(tensor_apply_2d, [3,3]; dependencies = "", bonus_quadorder = 0)
    return PDEOperator{Float64, APT_BilinearForm, AT}(name,[ϵ, ϵ], action, [1], 1, regions, store, AssemblyInitial)
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform a(u,v) = (C ϵ(u), ϵ(v)) where C is the 3D stiffness tensor
for isotropic media in Voigt notation, i.e.
ℂ ϵ(u) = 2 μ ϵ(u) + λ tr(ϵ(u)) for Lame parameters μ and λ

    In Voigt notation ℂ is a 6 x 6 matrix
    ℂ = [c11,c12,c12,  0,  0,  0
         c12,c11,c12,  0,  0,  0
         c12,c12,c11,  0,  0,  0
           0,  0,  0,c44,  0,  0
           0,  0,  0,  0,c44,  0
           0,  0,  0,  0,  0,c44]   

    where c44 = μ, c12 = λ and c11 = 2*c44 + c12

Note: ϵ is the symmetric part of the gradient (in Voigt notation)
    
"""
function HookStiffnessOperator3D(μ, λ;
    name = "(ℂ(μ,λ) ϵ(#A),ϵ(#T))", 
    AT::Type{<:AssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    ϵ = SymmetricGradient{1},
    store::Bool = false)

    @assert ϵ <: SymmetricGradient

    function tensor_apply_3d(result, input)
        result[1] = (λ + 2*μ)*input[1] + λ*(input[2] + input[3])
        result[2] = (λ + 2*μ)*input[2] + λ*(input[1] + input[3])
        result[3] = (λ + 2*μ)*input[3] + λ*(input[1] + input[2])
        result[4] = μ*input[4]
        result[5] = μ*input[5]
        result[6] = μ*input[6]
        return nothing
    end   
    action = Action(tensor_apply_3d, [6,6]; dependencies = "", bonus_quadorder = 0)
    return PDEOperator{Float64, APT_BilinearForm, AT}(name,[ϵ, ϵ], action, [1], 1, regions, store, AssemblyInitial)
end






"""
$(TYPEDSIGNATURES)

generates a BilinearForm defined by the following arguments:

- operators_linear  : operator for the two linear arguments (usually ansatz and test function)
- operators_current : additional operators for other unknowns
- coeff_from        : either PDE unknown ids or block ids for CurrentSolution given to assembly_operator! that should be used for operators_current
- action            : tells how to further combine the operators_current+operator_ansatz evaluations (=input of action) to a result that is multiplied with the test function operator
                      (if no action is specified, the full input vector is dot-producted with the test function operator evaluation)

Optional arguments:

- apply_action_to   : specifies which of the two linear arguments is part of the action input ([1] = ansatz, [2] = test)
- regions           : specifies in which regions the operator should assemble, default [0] means all regions
- name              : name for this BilinearForm that is used in print messages
- AT                : specifies on which entities of the grid the BilinearForm is assembled (default: ON_CELLS)
- APT               : specifies the subtype of the APT_BilinearForm AssemblyPattern used for assembly (e.g. for lumping (wip))
- factor            : additional factor that is multiplied during assembly
- transposed_assembly : transposes the resulting assembled matrix (consider true here for non-symmetric operators)
- also_transposed_block : when true toggles assembly of transposed system matrix block
- transpose_factor  : factor for transposed block (default = factor)
- store             : stores a matrix of the BilinearForm with the latest assembly result
                      (e.g. when the operators sits in a system block that has to be reassembled in an iterative scheme)

Details on the action:
The action is an Action consisting of a kernel function with interface (result, input, ...) and additional argument information. During assembly
input will be filled with the operator evaluations of the other unknowns (i.e. operator_current, if specified) and appended to that the operator
evaluation of one of the two linear argument (decided by apply_action_to). The result computed by the kernel function
is multiplied (dot product) with the operator evaluation of the other linear argument.
If no action is given, the assembly tries to multiply the operator evaluations (that would have been given as input) directly.
    
"""
function BilinearForm(
    operators_linear::Array{DataType,1},
    operators_current::Array{DataType,1},
    coeff_from::Array{Int,1},
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    APT::Type{<:APT_BilinearForm} = APT_BilinearForm,
    apply_action_to = [1],
    factor = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false,
    also_transposed_block = false,
    transpose_factor = factor,
    store::Bool = false)

    # check formalities
    @assert apply_action_to in [[1],[2]] "Action must be applied to [1] or [2]"

    # construct PDEoperator
    if name == "auto"
        if action.name == "no action"
            name = "($(operators_linear[2])(#A), $(operators_linear[1])(#T))"
        else
            name = apply_action_to == 1 ? "(A(...,$(operators_linear[1])(#A)), $(operators_linear[2])(#T))" : "($(operators_linear[1])(#A), A(...,$(operators_linear[2])(#T)))"
        end
        if factor == -1
            name = "-" * name
        elseif factor != 1
            name = "$factor " * name
        end
    end
    
    append!(operators_current, operators_linear)
    O = PDEOperator{Float64, APT, AT}(name, operators_current, action, apply_action_to, factor, regions, store, AssemblyAuto)
    O.fixed_arguments = 1:length(coeff_from)
    O.fixed_arguments_ids = coeff_from
    O.transposed_assembly = transposed_assembly
    O.transposed_copy = also_transposed_block
    O.transpose_factor = transpose_factor
    return O
end

"""
$(TYPEDSIGNATURES)

same as other constructor but with operators_current = [] (no other implicit dependencies)
"""
function BilinearForm(
    operators_linear::Array{DataType,1},
    action::AbstractAction = NoAction();
    kwargs...)

    BilinearForm(operators_linear, Array{DataType,1}([]), Array{Int,1}([]), action; kwargs...)
end

"""
$(TYPEDSIGNATURES)

constructs a convection term of the form c(a,u,v) = (a_operator(a)*ansatz_operator(u),test_operator(v))
as a BilinearForm (or NonlinearForm, see newton argument)

- a_from      : id of registered unknown to be used in the spot a
- a_operator  : operator applied to a
- xdim        : expected space dimension
- ncomponents : expected numer of components of a

optional arguments:

- newton          : generates a NonlinearForm instead of a BilinearForm that triggers assembly of Newton terms for c(u,u,v)
- a_to            : position of a argument, set a_to = 2 to trigger assembly of c(u,a,v)
- ansatz_operator : operator used in the spot u (default: Gradient)
- test_operator   : operator used in the spot v (default: Identity)
- factor          : additional factor multiplied in assemblxy (default: 1)
- regions         : specifies in which regions the operator should assemble, default [0] means all regions
- name            : name for this operator that is used in print messages
- AT              : specifies on which entities of the grid the operator is assembled (default: ON_CELLS)
- store           : stores a matrix of the operator with the latest assembly result

"""
function ConvectionOperator(
    a_from::Int, 
    a_operator::Type{<:AbstractFunctionOperator},
    xdim::Int,
    ncomponents::Int;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    a_to::Int = 1,
    factor = 1,
    ansatz_operator::Type{<:AbstractFunctionOperator} = Gradient,
    test_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0],
    newton::Bool = false,
    store::Bool = false,
    transposed_assembly::Bool = true,
    bonus_quadorder = 0)

    @assert a_to in [1,2] "a must go to position 1 or 2"

    # action input consists of two inputs
    # input[1:xdim] = operator1(a)
    # input[xdim+1:end] = grad(u)
    argsizes = [ncomponents, xdim + ncomponents*xdim]
    if a_to == 1 || newton # input = [a,u]
        function convection_function_fe_1(result, input)
            for j = 1 : ncomponents
                result[j] = 0
                for k = 1 : xdim
                    result[j] += input[k]*input[xdim+(j-1)*xdim+k]
                end
            end
            return nothing
        end    
        convection_action = Action(convection_function_fe_1,argsizes; dependencies = "", bonus_quadorder = bonus_quadorder)
    elseif fixed_argument == 2 # input = [u,a]
        function convection_function_fe_2(result, input)
            for j = 1 : ncomponents
                result[j] = 0
                for k = 1 : xdim
                    result[j] += input[xdim*ncomponents+k]*input[(j-1)*xdim+k]
                end
            end
            return nothing
        end    
        convection_action = Action(convection_function_fe_2,argsizes; dependencies = "", bonus_quadorder = bonus_quadorder)
    end
    if newton
        function convection_jacobian(jac, input)
            for j = 1 : ncomponents, k = 1 : xdim
                jac[j,k] = input[xdim+(j-1)*xdim+k]
                jac[j,xdim+(j-1)*xdim+k] = input[k]
            end
            return nothing
        end    
        ## generates a nonlinear form with automatic Newton operators by AD
        if name == "auto"
            name = "(($a_operator(#1) ⋅ $ansatz_operator) #1, $(test_operator)(#T))"
        end
        return NonlinearForm(test_operator, [a_operator, ansatz_operator], [a_from,a_from], convection_function_fe_1,argsizes; name = name, jacobian = convection_jacobian, bonus_quadorder = bonus_quadorder, store = store)     
    else
        ## returns linearised convection operators as a trilinear form (Picard iteration)
        if name == "auto"
            if a_to == 1
                name = "(($a_operator(#1) ⋅ $ansatz_operator) #A, $(test_operator)(#T))"
            elseif a_to == 2
                name = "(($a_operator(#A) ⋅ $ansatz_operator) #1, $(test_operator)(#T))"
            end
        end
        
        O = PDEOperator{Float64, APT_BilinearForm, AT}(name,[a_operator,ansatz_operator,test_operator], convection_action, [1,2], factor, regions, store, AssemblyAuto)
        O.fixed_arguments = [a_to]
        O.fixed_arguments_ids = [a_from]
        O.transposed_assembly = transposed_assembly
        return O
    end
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform a(u,v) = (beta x curl(u),v)
where beta is the id of some unknown vector field of the PDEDescription, u and v
are also vector-fields and x is the cross product (so far this is only implemented in 2D)
    
"""
function ConvectionRotationFormOperator(
    beta::Int,
    beta_operator::Type{<:AbstractFunctionOperator},
    xdim::Int, ncomponents::Int;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    factor = 1,
    ansatz_operator::Type{<:AbstractFunctionOperator} = Curl2D,
    test_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0])
    if xdim == 2
        # action input consists of two inputs
        # input[1:xdim] = operator1(a)
        # input[xdim+1:end] = curl(u)
        function rotationform_2d()
            function closure(result, input)
                result[1] = input[2] * input[3]
                result[2] = - input[1] * input[3]
            end    
        end    
        action_kernel = ActionKernel(rotationform_2d(),[2, 3]; dependencies = "", bonus_quadorder = 0)
        convection_action = Action{Float64}( action_kernel)
        if name == "auto"
            name = "((β × ∇) #A, #T)"
        end
        O = PDEOperator{Float64, APT_BilinearForm, AT}(name,[beta_operator,ansatz_operator,test_operator], convection_action, [1,2], factor, regions)
        O.fixed_arguments = [1]
        O.fixed_arguments_ids = [beta]
        O.transposed_assembly = true
        return O
    else
        @error "The rotation form of the convection operator is currently only available in 2D (in 3D please implement it yourself using BilinearForm and a user-defined action)"
    end
end

"""
$(TYPEDSIGNATURES)

generates a NonlinearForm defined by the following arguments:

- operator_test     : operator for the test function
- operators_current : additional operators for other unknowns
- coeff_from        : either PDE unknown ids or block ids for CurrentSolution given to assembly_operator! that should be used for operators_current
- action_kernel     : function of interface (result, input, ...) that computes the nonlinear quantity that should be multiplied with the testfunction operator
- argsizes          : dimensions of [result, input] of kernel function

Optional arguments:

- dependencies      : code String for additional dependencies of the kernel/jacobians (substring of "XTI")
- jacobian          : default = "auto" triggers automatic computation of jacobians by ForwardDiff, otherwise user can specify a function of interface (jacobian, input, ...) with matching dimensions and dependencies
- sparse_jacobian   : use sparsity detection and sparse matrixes for local jacobians ?
- regions           : specifies in which regions the operator should assemble, default [0] means all regions
- name              : name for this NonlinearForm that is used in print messages
- AT                : specifies on which entities of the grid the NonlinearForm is assembled (default: ON_CELLS)
- factor            : additional factor that is multiplied during assembly
- store             : stores a matrix of the discretised NonlinearForm with the latest assembly result
- bonus_quadorder   : increases the quadrature order in assembly accordingly (additional to usual quadorder based on used FESpaces)
  

Some details: Given some operator G(u), the Newton iteration reads DG u_next = DG u - G(u) which is added to the rest of the (linear) operators in the PDEDescription.
The local jacobians (= jacobians of the operator kernel) to build DG needed for this are computed by
automatic differentation (ForwardDiff). The user can also specify a jacobian kernel function by hand (which may improve assembly times).

For default dependencies both the kernel functions for the operator and its jacobian have to satisfy the interface

    function name(result,input,...)

where input is a vector of the operators of the solution and result is either what then is multiplied with operator2 of the testfunction (or the jacobian).

"""
function NonlinearForm(
    operator_test::Type{<:AbstractFunctionOperator},
    operators_current::Array{DataType,1},
    coeff_from::Array{Int,1},
    action_kernel, # should be a function of input (result, input) or matching the specified dependencies
    argsizes::Array{Int,1};
    name::String = "nonlinear form",
    AT::Type{<:AssemblyType} = ON_CELLS,
    newton::Bool = true, # prepare operators for Newton algorithm
    sparse_jacobian = true, # use sparsity detection and sparse matrixes for local jacobians
    jacobian = "auto", # by automatic ForwarDiff, or should be a function of input (jacobian, input) with sizes matching the specified dependencies
    dependencies = "",
    bonus_quadorder::Int = 0,
    store::Bool = false,
    factor = 1,
    regions = [0])


    if length(argsizes) == 2
        push!(argsizes, argsizes[2])
    end

    operators = copy(operators_current)
    append!(operators, [operator_test])

    if newton
        if jacobian == "auto"
            name = name * " [AD-Newton]"
            jac_handler = OperatorWithADJacobian(action_kernel, argsizes; dependencies = dependencies, bonus_quadorder = bonus_quadorder, sparse_jacobian = sparse_jacobian)
        else
            name = name * " [Newton]"
            jac_handler = OperatorWithUserJacobian(action_kernel, jacobian, argsizes; dependencies = dependencies, bonus_quadorder = bonus_quadorder, sparse_jacobian = sparse_jacobian)
        end

        O = PDEOperator{Float64, APT_NonlinearForm, AT}(name, operators, jac_handler, 1:(length(coeff_from)), 1, regions)
        O.fixed_arguments = 1:length(coeff_from)
        O.fixed_arguments_ids = coeff_from
        O.newton_arguments = 1 : length(coeff_from) # if depended on different ids, operator is later splitted into several operators where newton_arguments refer only to subset
        O.factor = factor
        O.store_operator = store
        O.transposed_assembly = true
    else
        @error "currently only newton = true is possible"
    end
    return O
end


"""
$(TYPEDSIGNATURES)

generates a LinearForm L(v) = (f,operator(v)) from an action

- operator : operator applied to test function
- action   : action that computes a result to be multiplied with test function operator

Optional arguments:

- regions           : specifies in which regions the operator should assemble, default [0] means all regions
- name              : name for this LinearForm that is used in print messages
- AT                : specifies on which entities of the grid the LinearForm is assembled (default: ON_CELLS)
- factor            : additional factor that is multiplied during assembly
- store             : stores a vector of the discretised LinearForm with the latest assembly result

Details on the action:
The action is an Action consisting of a kernel function with interface (result, input, ...) and additional argument information. During assembly
input is ignored (only in this constructor for LinearForms). The result computed by the kernel function
is multiplied (dot product) with the operator evaluation of the test function.
"""
function LinearForm(
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    factor = 1,
    store::Bool = false)

    if name == "auto"
        name = "(A($operator(#T)), 1)"
    end
    O = PDEOperator{Float64, APT_LinearForm, AT}(name, [operator], action, [1], 1, regions, store, AssemblyAuto)
    O.factor = factor
    return O 
end


"""
$(TYPEDSIGNATURES)

generates a LinearForm L(v) = (f,operator(v)) from a DataFunction

- operator : operator applied to test function
- action   : DataFunction, evaluation is multiplied with test function operator

Optional arguments:

- regions           : specifies in which regions the operator should assemble, default [0] means all regions
- name              : name for this LinearForm that is used in print messages
- AT                : specifies on which entities of the grid the LinearForm is assembled (default: ON_CELLS)
- factor            : additional factor that is multiplied during assembly
- store             : stores a vector of the discretised LinearForm with the latest assembly result

"""
function LinearForm(
    operator::Type{<:AbstractFunctionOperator},
    f::AbstractUserDataType;
    name = "auto",
    kwargs...)

    if name == "auto"
        name = "($(f.name), $operator(#T))"
    end

    return LinearForm(operator, fdot_action(f); name = name, kwargs...)
end


"""
$(TYPEDSIGNATURES)

Creates a (PDE description level) LinearForm based on:

- operator_test     : operator for the test function (assumes linearity for that part)
- operators_current : additional operators for other unknowns
- coeff_from        : either PDE unknown ids or block ids for CurrentSolution given to assembly_operator! that should be used for operators_current
- action            : an Action with kernel of interface (result, input, kwargs) that takes input (= all but last operator evaluations) and computes result to be dot-producted with test function evaluation
                      (if no action is specified, the full input vector is dot-producted with the test function operator evaluation)

Optional arguments:

- regions: specifies in which regions the operator should assemble, default [0] means all regions
- name : name for this LinearForm that is used in print messages
- AT : specifies on which entities of the grid the LinearForm is assembled (default: ON_CELLS)
- factor : additional factor that is multiplied during assembly
- store : stores a vector of the LinearForm with the latest assembly result
  (e.g. when the operators sits in a system block that has to be reassembled in an iterative scheme)
    
Details on the action:
The action is an Action consisting of a kernel function with interface (result, input, ...) and additional argument information. During assembly
input will be filled with the operator evaluations of the other unknowns (i.e. operators_current). The result computed by the kernel function
is multiplied (dot product) with the operator evaluation of the test function (i.e. operator_test).
If no action is given, the assembly tries to multiply the operator evaluations (that would have been given as input) directly.
"""
function LinearForm(
    operator_test::Type{<:AbstractFunctionOperator},                   # operator to be evaluted for test function
    operators_current::Array{DataType,1} = [],                         # operators to be evaluated for current solution (in order of expected action input)
    coeff_from::Array{Int,1} = ones(Int, length(operators_current)),   # unknown ids for operators_current
    action::AbstractAction = NoAction();                               # action that takes all input operators and computes some result that is multiplied with test function operator
    regions::Array{Int,1} = [0],
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    factor = 1,
    store::Bool = false)

    if name == "auto"
        name = "($(action.name), $operator_test(#T))"
    end

    push!(operators_current, operator_test)
    O = PDEOperator{Float64, APT_LinearForm, AT}(name, operators_current, action, 1:(length(coeff_from)), 1, regions, store, AssemblyAuto)
    O.fixed_arguments = 1:length(coeff_from)
    O.fixed_arguments_ids = coeff_from
    O.factor = factor
    return O
end




"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = ((β ⋅ ∇) u,v) with some user-specified DataFunction β.
The user also has to specify the number of components (ncomponents) the convection is applied to.
The operators for u and v can be changed (if this leads to something reasonable).
    
"""
function ConvectionOperator(
    β::AbstractUserDataType,
    ncomponents::Int; 
    name = "auto", 
    store::Bool = false,
    AT::Type{<:AssemblyType} = ON_CELLS,
    ansatz_operator::Type{<:AbstractFunctionOperator} = Gradient,
    test_operator::Type{<:AbstractFunctionOperator} = Identity,
    transposed_assembly::Bool = true,
    regions::Array{Int,1} = [0])

    T = Float64
    xdim = β.argsizes[1]

    function convection_kernel(result, input, kwargs...)
        # evaluate β
        eval_data!(β, kwargs...)
        # compute (β ⋅ ∇) u
        for j = 1 : ncomponents
            result[j] = 0.0
            for k = 1 : xdim
                result[j] += β.val[k]*input[(j-1)*xdim+k]
            end
        end
    end    
    action = Action(convection_kernel, [ncomponents, ncomponents*xdim]; dependencies = dependencies(β), bonus_quadorder = β.bonus_quadorder)
    
    if name == "auto"
        name = "((β ⋅ $(ansatz_operator)) #A, $test_operator(#T))"
    end

    O = PDEOperator{T, APT_BilinearForm, AT}(name, [ansatz_operator, test_operator], action, [1], 1, regions, store, AssemblyAuto)
    O.transposed_assembly = transposed_assembly
    return O
end



function update_storage!(O::PDEOperator, SC, CurrentSolution::FEVector{T,Tv,Ti}, j::Int, k::Int, o::Int; factor = 1, time::Real = 0) where {T,Tv,Ti}
    @logmsg MoreInfo "Updating storage of PDEOperator $(O.name) in LHS block [$j,$k] (on thread $(Threads.threadid()))"

    set_time!(O.action, time)
    APT = typeof(O).parameters[2]
    AT = typeof(O).parameters[3]
    if APT <: APT_BilinearForm
        FES = Array{FESpace{Tv,Ti},1}(undef, 2)
        FES[1] = CurrentSolution[j].FES
        FES[2] = CurrentSolution[k].FES
    elseif APT <: APT_NonlinearForm
        FES = Array{FESpace{Tv,Ti},1}(undef, length(O.fixed_arguments))
        for a = 1 : length(O.fixed_arguments)
            FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
        end
        push!(FES, CurrentSolution[j].FES) # testfunction always refers to matrix row in this pattern !!!
    else
        @error "No storage functionality available for this operator!"
    end
    skip_preps = true
    if APT <: APT_NonlinearForm
        if !O.storage_init || typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            SC.LHS_AssemblyPatterns[j,k][o] = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
            SC.LHS_AssemblyPatterns[j,k][o].newton_args = O.newton_arguments
            skip_preps = false
            A = FEMatrix{T}(FES[1], FES[end])
            b = FEVector{T}(FES[1])
            #set_nonzero_pattern!(A)
        else
            A = O.storage_A
            b = O.storage_b
            if size(A[1,1],1) < FES[1].ndofs || size(A[1,1],2) < FES[end].ndofs
                @info "re-init storage"
                A = FEMatrix{T}(FES[1], FES[end])
                SC.LHS_AssemblyPatterns[j,k][o] = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
                SC.LHS_AssemblyPatterns[j,k][o].newton_args = O.newton_arguments
                #set_nonzero_pattern!(A)
                skip_preps = false
                b = zeros(T,FES[1].ndofs)
            else
                fill!(A[1,1],0)
                fill!(b.entries,0)
            end
        end
        
        full_assemble!(A[1,1], b[1], SC.LHS_AssemblyPatterns[j,k][o], CurrentSolution[O.fixed_arguments_ids]; transposed_assembly = O.transposed_assembly, factor = factor, skip_preps = skip_preps)
        flush!(A.entries)
        O.storage_A = A
        O.storage_b = b
    else
        if !O.storage_init || typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            A = FEMatrix{T}(FES[1], FES[2])
            SC.LHS_AssemblyPatterns[j,k][o] = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
            O.storage_init = true
            skip_preps = false
        else
            A = O.storage_A
            if size(A[1,1],1) < FES[1].ndofs || size(A[1,1],2) < FES[2].ndofs
                A = FEMatrix{T}(FES[1], FES[2])
                SC.LHS_AssemblyPatterns[j,k][o] = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
                skip_preps = false
            else
                fill!(A[1,1],0)
            end
        end
        assemble!(A[1,1], SC.LHS_AssemblyPatterns[j,k][o]; transposed_assembly = O.transposed_assembly, factor = factor, skip_preps = skip_preps)
        flush!(A.entries)
        O.storage_A = A
    end
    return SC.LHS_AssemblyPatterns[j,k][o].last_allocations
end

function update_storage!(O::PDEOperator, SC, CurrentSolution::FEVector{T,Tv,Ti}, j::Int, o::Int; factor = 1, time::Real = 0) where {T,Tv,Ti}

    @logmsg MoreInfo "Updating storage of PDEOperator $(O.name) in RHS block [$j] (on thread $(Threads.threadid()))"

    set_time!(O.action, time)
    APT = typeof(O).parameters[2]
    AT = typeof(O).parameters[3]
    if APT <: APT_LinearForm
        FES = Array{FESpace{Tv,Ti},1}(undef, length(O.fixed_arguments))
        for a = 1 : length(O.fixed_arguments)
            FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
        end
        push!(FES, CurrentSolution[j].FES) # testfunction always refers to matrix row in this pattern !!!
    else
        @error "No storage functionality available for this operator!"
    end
    if !O.storage_init
        O.storage_b = zeros(T,FES[1].ndofs)
        O.storage_init = true
    else
        if length(O.storage_b) < FES[1].ndofs
            O.storage_b = zeros(T,FES[1].ndofs)
        else
            fill!(O.storage_b,0)
        end
    end
    Pattern = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action_rhs,O.apply_action_to,O.regions)
    assemble!(O.storage_b, Pattern, CurrentSolution[O.fixed_arguments_ids]; factor = factor, skip_preps = false)
    return Pattern.last_allocations
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock{TvM,TiM,TvG,TiG}, CurrentSolution; fixed = 0) where{T,TvM,TiM,TvG,TiG,APT<:APT_BilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    if length(O.operators4arguments) - 1 == length(O.fixed_arguments_ids)
        @warn "PDEOperator was recasted into LinearForm pattern before, trying to cope with it..."
        O.fixed_arguments = O.fixed_arguments[1:end-1]
        O.fixed_arguments_ids = O.fixed_arguments_ids[1:end-1]
        if O.active_recast
            # reswitch last two operators
            nops = length(O.operators4arguments)
            O.operators4arguments[[nops,nops-1]] = O.operators4arguments[[nops-1,nops]]
        end
    end
    FES = Array{FESpace{TvG,TiG},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES, O.transposed_assembly ? A.FESY : A.FESX)
    push!(FES, O.transposed_assembly ? A.FESX : A.FESY)
    return AssemblyPattern{APT, TvM, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

# recasting of BilinearForm into LinearForm if assembled into a vector
function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock{TvV,TvG,TiG}, CurrentSolution; fixed = 0, fixed_id = 0) where {T,TvV,TvG,TiG,APT<:APT_BilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    @assert fixed in [1,2] && fixed_id > 0 "need to fix one linear argument in the bilinearform to assemble into vector (specify fixed = 1 or 2 and fixed_id)!"
    if length(O.operators4arguments) - 1 == length(O.fixed_arguments_ids)
        @warn "PDEOperator was recasted into LinearForm pattern before, trying to cope with it..."
        if O.active_recast
            # reswitch last two operators
            nops = length(O.operators4arguments)
            O.operators4arguments[[nops,nops-1]] = O.operators4arguments[[nops-1,nops]]
        end
        O.fixed_arguments[end] = length(O.fixed_arguments) + 1
        O.fixed_arguments_ids[end] = fixed_id
    else
        push!(O.fixed_arguments, length(O.fixed_arguments) + 1)
        push!(O.fixed_arguments_ids, fixed_id)
    end
    FES = Array{FESpace{TvG,TiG},1}(undef, 2)
    FES = Array{FESpace{TvG,TiG},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES, b.FES)
    if fixed == 2 # (fixed == 2 && O.apply_action_to == [1]) || (fixed == 1 && O.apply_action_to == [2])   
        # switch last two operators
        nops = length(O.operators4arguments)
        O.operators4arguments[[nops,nops-1]] = O.operators4arguments[[nops-1,nops]]
        O.active_recast = true # remember that operators have been switched
    end
    return AssemblyPattern{APT_LinearForm, TvV, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock{TvV,TvG,TiG}, CurrentSolution; fixed = 0, fixed_id = 0) where{T,TvV,TvG,TiG,APT<:APT_LinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES, b.FES) # testfunction always refers to matrix row in this pattern !!!
    return AssemblyPattern{APT, TvV, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

"""
$(TYPEDSIGNATURES)

assembles the operator O into the given FEMatrixBlock A using FESpaces from A.
"""
function assemble_operator!(A::FEMatrixBlock, O::PDEOperator, CurrentSolution::Union{Nothing,FEVector} = nothing; Pattern = nothing, skip_preps::Bool = false, time::Real = 0, At = nothing)
    if Pattern === nothing
        Pattern = create_assembly_pattern(O, A, CurrentSolution)
    end
    set_time!(O.action, time)
    if At !== nothing
        assemble!(A, Pattern; skip_preps = skip_preps, transposed_assembly = O.transposed_assembly, factor = O.factor, transpose_copy = At)
    else
        if length(O.fixed_arguments_ids) > 0
            assemble!(A, Pattern, CurrentSolution[O.fixed_arguments_ids], skip_preps = skip_preps, transposed_assembly = O.transposed_assembly, factor = O.factor, fixed_arguments = O.fixed_arguments)
        else
            assemble!(A, Pattern, skip_preps = skip_preps, transposed_assembly = O.transposed_assembly, factor = O.factor)
        end
    end
    flush!(A.entries)
end


function assemble_operator!(b::FEVectorBlock, O::PDEOperator, CurrentSolution = nothing; Pattern = nothing, skip_preps::Bool = false, factor = 1, time::Real = 0, fixed = 0, fixed_id = 0)
    if Pattern === nothing
        Pattern = create_assembly_pattern(O, b, CurrentSolution; fixed = fixed, fixed_id = fixed_id)
    end
    set_time!(O.action, time)
    if length(O.fixed_arguments_ids) > 0
        assemble!(b, Pattern, CurrentSolution[O.fixed_arguments_ids], skip_preps = skip_preps, factor = O.factor*factor, fixed_arguments = O.fixed_arguments)
    else
        assemble!(b, Pattern, skip_preps = skip_preps, factor = O.factor)
    end
end






function assemble!(A::FEMatrixBlock, SC, j::Int, k::Int, o::Int, O::PDEOperator, CurrentSolution::FEVector; time::Real = 0, At = nothing)  
    if O.store_operator == true
        @logmsg DeepInfo "Adding PDEOperator $(O.name) from storage"
        addblock!(A,O.storage_A[1,1]; factor = O.factor)
        if At !== nothing
            addblock!(At, O.storage_A[1,1]; factor = O.factor, transpose = true)
        end
    else
        ## find assembly pattern
        skip_preps = true
        if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            SC.LHS_AssemblyPatterns[j,k][o] = create_assembly_pattern(O, A, CurrentSolution)
            skip_preps = false
        end
        ## assemble
        set_time!(O.action, time)
        assemble_operator!(A, O, CurrentSolution; Pattern = SC.LHS_AssemblyPatterns[j,k][o], time = time, At = At, skip_preps = skip_preps)
    end
end


function assemble!(b::FEVectorBlock, SC, j::Int, o::Int, O::PDEOperator, CurrentSolution::FEVector; factor = 1, time::Real = 0)
    if O.store_operator == true
        @logmsg DeepInfo "Adding PDEOperator $(O.name) from storage"
        addblock!(b, O.storage_b; factor = factor * O.factor)
    else
        ## find assembly pattern
        skip_preps = true
        if typeof(SC.RHS_AssemblyPatterns[j][o]).parameters[1] <: APT_Undefined
            SC.RHS_AssemblyPatterns[j][o] = create_assembly_pattern(O, b, CurrentSolution)
            skip_preps = false
        end

        ## assemble
        set_time!(O.action_rhs, time)
        assemble_operator!(b, O, CurrentSolution; Pattern = SC.RHS_AssemblyPatterns[j][o], time = time, skip_preps = skip_preps, factor = factor)
    end
end


# LHS operator is assembled to RHS block (due to subiteration configuration)
function assemble!(b::FEVectorBlock, SC, j::Int, k::Int, o::Int, O::PDEOperator, CurrentSolution::FEVector; factor = 1, time::Real = 0, fixed_component = 0)
    if O.store_operator == true
        @logmsg DeepInfo "Adding PDEOperator $(O.name) from storage"
        addblock_matmul!(b,O.storage_A[1,1],CurrentSolution[fixed_component]; factor = factor * O.factor)
    else
        ## find assembly pattern
        skip_preps = true
        if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            fixed = (j == fixed_component) * (O.apply_action_to == 1) ? 1 : 2
            SC.LHS_AssemblyPatterns[j,k][o] = create_assembly_pattern(O, b, CurrentSolution; fixed = fixed, fixed_id = fixed_component)
            skip_preps = false
        end

        ## assemble
        set_time!(O.action, time)
        if length(O.fixed_arguments_ids) > 0
            assemble!(b, SC.LHS_AssemblyPatterns[j,k][o],CurrentSolution[O.fixed_arguments_ids]; skip_preps = skip_preps, factor = O.factor*factor, fixed_arguments = O.fixed_arguments)
        else
            assemble!(b, SC.LHS_AssemblyPatterns[j,k][o]; skip_preps = skip_preps, factor = O.factor*factor)
        end
    end
end



#####################################################
### ASSEMBLY/EVALUATION OF NONLINEAR PDEOPERATORS ###
#####################################################


function full_assemble_operator!(A::FEMatrixBlock, b::FEVectorBlock, O::PDEOperator, CurrentSolution::Union{Nothing,FEVector} = nothing; Pattern = nothing, skip_preps::Bool = false, time::Real = 0, At = nothing)
    if Pattern === nothing
        Pattern = create_assembly_pattern(O, A, CurrentSolution)
    end
    set_time!(O.action, time)
    if length(O.fixed_arguments_ids) > 0
        full_assemble!(A, b, Pattern, CurrentSolution[O.fixed_arguments_ids]; skip_preps = skip_preps, transposed_assembly = O.transposed_assembly, factor = O.factor)
    else
        #full_assemble!(A, b, Pattern; skip_preps = skip_preps, factor = O.factor)
    end
    flush!(A.entries)
end

function full_assemble!(A::FEMatrixBlock, b::FEVectorBlock, SC, j::Int, k::Int, o::Int, O::PDEOperator, CurrentSolution::FEVector; time::Real = 0, At = nothing)  
    if O.store_operator == true
        @logmsg DeepInfo "Adding PDEOperator $(O.name) from storage"
        addblock!(A,O.storage_A[1,1]; factor = O.factor)
        addblock!(b,O.storage_b[1]; factor = O.factor)
    else
        ## find assembly pattern
        skip_preps = true
        if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            SC.LHS_AssemblyPatterns[j,k][o] = create_assembly_pattern(O, A, CurrentSolution)
            skip_preps = false
        end
        ## assemble
        set_time!(O.action, time)
        full_assemble_operator!(A, b, O, CurrentSolution; Pattern = SC.LHS_AssemblyPatterns[j,k][o], time = time, At = At, skip_preps = skip_preps)
    end
end
function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock{TvM,TiM,TvG,TiG}, CurrentSolution::FEVector) where{T,TvM,TiM,TvG,TiG,APT<:APT_NonlinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES,A.FESX) # testfunction always refers to matrix row in this pattern !!!
    AP = AssemblyPattern{APT, TvM, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
    AP.newton_args = O.newton_arguments
    return AP
end

function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock{TvV,TvG,TiG}, CurrentSolution::FEVector) where{T,TvV,TvG,TiG,APT<:APT_NonlinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES,b.FES)
    return AssemblyPattern{APT, TvV, AT}(O.name, FES, O.operators4arguments,O.action_rhs,O.apply_action_to,O.regions)
end

function evaluate(O::PDEOperator{T,APT,AT}, CurrentSolution::FEVector{T,Tv,Ti}, TestFunctionBlock::FEVectorBlock{T,Tv,Ti}; factor = 1, time::Real = 0) where {T, Tv, Ti, APT, AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{Tv,Ti},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES,TestFunctionBlock.FES)
    AP = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action_eval,O.apply_action_to,O.regions)

    ## assemble
    set_time!(O.action_rhs, time)
    return evaluate(AP, CurrentSolution[O.fixed_arguments_ids], TestFunctionBlock; skip_preps = false, factor = factor)
end

function eval_assemble!(b::FEVectorBlock{T,Tv,Ti}, O::PDEOperator{T,APT,AT}, CurrentSolution::FEVector{T,Tv,Ti}; factor = 1, time::Real = 0) where {T, Tv, Ti, APT, AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{Tv,Ti},1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES,b.FES)
    AP = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action_eval,O.apply_action_to,O.regions)

    ## assemble
    set_time!(O.action_rhs, time)
    return assemble!(b, AP, CurrentSolution[O.fixed_arguments_ids]; skip_preps = false, factor = factor)
end



##############################
###### OTHER OPERATORS #######
##############################

struct DiagonalOperator{T} <: AbstractPDEOperator
    name::String
    value::T
    onlyz::Bool
    regions::Array{Int,1}
end

"""
$(TYPEDSIGNATURES)

puts _value_ on the diagonal entries of the cell dofs within given _regions_

if _onlyz_ == true only values that are zero are changed

can only be applied in PDE LHS
"""
function DiagonalOperator(value::Real = 1.0; name = "auto", onlynz::Bool = true, regions::Array{Int,1} = [0])
    if name == "auto"
        name = "diag($value)"
    end
    return DiagonalOperator{typeof(value)}(name, value, onlynz, regions)
end


function assemble!(A::FEMatrixBlock{TvM,TiM,TvG,TiG}, SC, j::Int, k::Int, o::Int,  O::DiagonalOperator{T}, CurrentSolution; time::Real = 0) where {T,TvM,TiM,TvG,TiG}
    @debug "Assembling DiagonalOperator $(O.name)"
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xCellDofs::DofMapTypes{TiG} = FE1[CellDofs]
    xCellRegions::GridRegionTypes{TiG} = FE1.xgrid[CellRegions]
    ncells::Int = num_sources(xCellDofs)
    dof::Int = 0
    value::T = O.value
    Am::ExtendableSparseMatrix{TvM,TiM} = A.entries
    for item = 1 : ncells
        for r = 1 : length(O.regions) 
            # check if item region is in regions
            if xCellRegions[item] == O.regions[r] || O.regions[r] == 0
                for k = 1 : num_targets(xCellDofs,item)
                    dof = xCellDofs[k,item] + A.offsetX
                    if O.onlyz == true
                        if Am[dof,dof] == zero(TvM)
                           _addnz(A.entries,dof,dof,value,1)
                        end
                    else
                        _addnz(A.entries,dof,dof,value,1)
                    end    
                end
            end
        end
    end
end


struct CopyOperator <: AbstractPDEOperator
    name::String
    copy_from::Int
    factor::Real
end

"""
$(TYPEDSIGNATURES)

copies entries from TargetVector to rhs block

can only be applied in PDE RHS
"""
function CopyOperator(copy_from, factor)
    return CopyOperator("CopyOperator",copy_from, factor)
end


function assemble!(b::FEVectorBlock, SC, j::Int, o::Int, O::CopyOperator, CurrentSolution::FEVector; time::Real = 0) 
    for j = 1 : length(b)
        b[j] = CurrentSolution[O.copy_from][j] * O.factor
    end
end


function check_PDEoperator(O::CopyOperator, involved_equations)
    return true, true, AssemblyAlways
end



mutable struct FVConvectionDiffusionOperator{T} <: AbstractPDEOperator
    name::String
    μ::T               # diffusion coefficient
    beta_from::Int           # component that determines
    fluxes::Array{T,2} # saves normalfluxes of beta here
end


"""
$(TYPEDSIGNATURES)

 finite-volume convection diffusion operator (for cell-wise P0 rho)

 - div(μ ∇ρ + β ρ)

 For μ = 0, the upwind divergence div_upw(β ρ) is generated
 For μ > 0, TODO
                   
"""
function FVConvectionDiffusionOperator(beta_from::Int; μ = 0.0)
    FVConvectionDiffusionOperator{Float64}(beta_from; μ = μ)
end
function FVConvectionDiffusionOperator{T}(beta_from::Int; μ::T = 0.0) where {T}
    @assert beta_from > 0
    fluxes = zeros(T,0,1)
    return FVConvectionDiffusionOperator("FVConvectionDiffusion",μ,beta_from,fluxes)
end


function check_PDEoperator(O::FVConvectionDiffusionOperator, involved_equations)
    return O.beta_from in involved_equations, false, AssemblyAlways
end

function check_dependency(O::FVConvectionDiffusionOperator, arg::Int)
    return O.beta_from == arg
end


#= 

(1) calculate normalfluxes from component at _beta_from_
(2) compute FV flux on each face and put coefficients on neighbouring cells in matrix

    if kappa == 0:
          div_upw(beta*rho)|_T = sum_{F face of T} normalflux(F) * rho(F)

          where rho(F) is the rho in upwind direction 

    and put it into P0xP0 matrix block like this:

          Loop over cell, face of cell

              other_cell = other face neighbour cell
              if flux := normalflux(F_j) * CellFaceSigns[face,cell] > 0
                  A(cell,cell) += flux
                  A(other_cell,cell) -= flux
              else
                  A(other_cell,other_cell) -= flux
                  A(cell,other_cell) += flux

=#
function assemble!(A::FEMatrixBlock, SC, j::Int, k::Int, o::Int,  O::FVConvectionDiffusionOperator{T}, CurrentSolution::FEVector{T,Tv,Ti}; time::Real = 0) where {T,Tv,Ti}
    @logmsg MoreInfo "Assembling FVConvectionOperator $(O.name) into matrix"
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xFaceNodes::Union{VariableTargetAdjacency{Ti},Array{Ti,2}} = FE1.xgrid[FaceNodes]
    xFaceNormals::Array{Tv,2} = FE1.xgrid[FaceNormals]
    xFaceCells::Union{VariableTargetAdjacency{Ti},Array{Ti,2}} = FE1.xgrid[FaceCells]
    xFaceVolumes::Array{Tv,1} = FE1.xgrid[FaceVolumes]
    xCellFaces::Union{VariableTargetAdjacency{Ti},Array{Ti,2}} = FE1.xgrid[CellFaces]
    xCellFaceSigns::Union{VariableTargetAdjacency{Ti},Array{Ti,2}} = FE1.xgrid[CellFaceSigns]
    nfaces::Int = num_sources(xFaceNodes)
    ncells::Int = num_sources(xCellFaceSigns)
    nnodes::Int = num_sources(FE1.xgrid[Coordinates])
    
    # ensure that flux field is long enough
    if length(O.fluxes) < nfaces
        O.fluxes = zeros(T,1,nfaces)
    end
    # compute normal fluxes of component beta
    c::Int = O.beta_from
    fill!(O.fluxes,0)
    if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
        @debug "Creating assembly pattern for FV convection fluxes $(O.name)"
        SC.LHS_AssemblyPatterns[j,k][o] = ItemIntegrator([NormalFlux]; name = "u ⋅ n", T = T, AT = ON_FACES)
        evaluate!(O.fluxes,SC.LHS_AssemblyPatterns[j,k][o],CurrentSolution[c], skip_preps = false)
    else
        evaluate!(O.fluxes,SC.LHS_AssemblyPatterns[j,k][o],CurrentSolution[c], skip_preps = true)
    end

    fluxes::Array{T,2} = O.fluxes
    nfaces4cell::Int = 0
    face::Int = 0
    flux::T = 0.0
    other_cell::Int = 0
    for cell = 1 : ncells
        nfaces4cell = num_targets(xCellFaces,cell)
        for cf = 1 : nfaces4cell
            face = xCellFaces[cf,cell]
            other_cell = xFaceCells[1,face]
            if other_cell == cell
                other_cell = xFaceCells[2,face]
            end
            flux = fluxes[face] * xCellFaceSigns[cf,cell] # sign okay?
            if (other_cell > 0) 
                flux *= 1 // 2 # because it will be accumulated on two cells
            end       
            if flux > 0 # flow from cell to other_cell
                _addnz(A,cell,cell,flux,1)
                if other_cell > 0
                    _addnz(A,other_cell,cell,-flux,1)
                    _addnz(A,other_cell,other_cell,1e-16,1) # add zero to keep pattern for LU
                    _addnz(A,cell,other_cell,1e-16,1) # add zero to keep pattern for LU
                    # otherwise flow goes out of domain
                end    
            else # flow from other_cell into cell
                _addnz(A,cell,cell,1e-16,1) # add zero to keep pattern for LU
                if other_cell > 0 # flow comes from neighbour cell
                    _addnz(A,other_cell,other_cell,-flux,1)
                    _addnz(A,cell,other_cell,flux,1)
                    _addnz(A,other_cell,cell,1e-16,1) # add zero to keep pattern for LU
                else # flow comes from outside domain
                   #  A[cell,cell] += flux
                end 
            end
        end
    end
end


#####################################
##### CUSTOM MATRIX PDEOPERATOR #####
#####################################


struct CustomMatrixOperator{MT<:AbstractMatrix, T <: Real} <: AbstractPDEOperator
    name::String
    factor::T
    matrix::MT
    nonlinear::Bool
end

function CustomMatrixOperator(A; factor = 1.0, name = "custom matrix", nonlinear::Bool = false)
    return CustomMatrixOperator{typeof(A),typeof(factor)}(name, factor, A, nonlinear)
end

function check_PDEoperator(O::CustomMatrixOperator, involved_equations)
    return O.nonlinear, false, AssemblyAlways
end


function assemble!(A::FEMatrixBlock, SC, j::Int, k::Int, o::Int,  O::CustomMatrixOperator, CurrentSolution::FEVector; time = 0, At = nothing)
    @logmsg DeepInfo "Adding CustomMatrixOperator $(O.name)"
    addblock!(A,O.matrix; factor = O.factor)
    if At !== nothing
        addblock!(At,O.matrix; factor = O.factor, transpose = true)
    end
end


function assemble!(b::FEVectorBlock, SC, j::Int, o::Int, O::CustomMatrixOperator, CurrentSolution::FEVector; factor = 1, time::Real = 0)
    @logmsg DeepInfo "Adding CustomMatrixOperator $(O.name)"
    addblock!(b, O.matrix; factor = factor * O.factor)
end

# LHS operator is assembled to RHS block (due to subiteration configuration)
function assemble!(b::FEVectorBlock, SC, j::Int, k::Int, o::Int, O::CustomMatrixOperator, CurrentSolution::FEVector; factor = 1, time::Real = 0, fixed_component = 0)
    @logmsg DeepInfo "Adding CustomMatrixOperator $(O.name)"
    addblock_matmul!(b,O.matrix,CurrentSolution[fixed_component]; factor = factor * O.factor)
end


#####################################
##### CUSTOM MATRIX PDEOPERATOR #####
#####################################

mutable struct SchurComplement{T} <: AbstractPDEOperator
    name::String
    operatorA::AbstractPDEOperator
    operatorB::AbstractPDEOperator
    operatorC::AbstractPDEOperator # currently ignored and B = C assumed
    operatorf::AbstractPDEOperator
    ids::Array{Int,1}              # ids in PDEDescription
    factor::T
    zero_boundary::Bool
    nonlinear::Bool
    timedependent::Bool
    storage_ready::Bool
    last_ndofs::Int
    storage_LHS::AbstractMatrix
    storage_RHS::AbstractVector
    SchurComplement(A,B,C,f,ids; name = "Schur complement", nonlinear::Bool = false, timedependent::Bool = false, zero_boundary::Bool = true) = new{Float64}(name, A,B,C,f, ids, 1, zero_boundary, nonlinear, timedependent, false,0)
end


function check_PDEoperator(O::SchurComplement, involved_equations)
    return O.nonlinear, O.timedependent, AssemblyAlways
end

function assembleSchurComplement(SC, O::SchurComplement{Tv}, CurrentSolution::FEVector; time = 0) where{Tv}
    if O.storage_ready == false || O.last_ndofs != CurrentSolution[O.ids[1]].FES.ndofs
        Ti = Int64
        @logmsg DeepInfo "Assembling $(O.name)"
        @assert length(O.ids) == 2
        FES = [CurrentSolution[O.ids[1]].FES,CurrentSolution[O.ids[2]].FES]
        MA = FEMatrix{Tv}("sub-block A", FES[1])
        MB = FEMatrix{Tv}("sub-block B", FES[1],FES[2])
    # MC = FEMatrix{Tv}("sub-block C", FES[2],FES[1])
        vb = FEVector{Tv}("sub-block f", FES[1])
        ndofs1::Int = FES[1].ndofs
        ndofs2::Int = FES[2].ndofs
        # todo: allow several operators in each block, use storage if available
        assemble_operator!(MA[1,1], O.operatorA, CurrentSolution; time = time)
        assemble_operator!(MB[1,1], O.operatorB, CurrentSolution; time = time)
    #  assemble_operator!(MC[1,1], O.operatorC, CurrentSolution; time = time)
        assemble_operator!(vb[1], O.operatorf, CurrentSolution; time = time)
        flush!(MA.entries)
        flush!(MB.entries)

        ## erase fixed dofs on boundary
        cscmat::SparseMatrixCSC{Tv,Ti} = MB.entries.cscmatrix
        rows::Array{Ti,1} = rowvals(cscmat)
        valsB::Array{Tv,1} = cscmat.nzval
        value::Tv = 0

        if O.zero_boundary
            xBFaceDofs = FES[1][BFaceDofs]
            dof::Int = 0
            for j = 1 : num_sources(xBFaceDofs)
                for k = 1 : num_targets(xBFaceDofs,j)
                    dof = xBFaceDofs[j,k]
                    MA.entries[dof,dof] = 1e60
                end
            end
        end

        diagA::Array{Tv,1} = diag(MA.entries)

        S = ExtendableSparseMatrix{Tv,Ti}(ndofs2,ndofs2)
        CinvAf = zeros(Tv,ndofs2)
        # compute S = B' inv(A_diag) B (currently assuming that B = C)
        for i = 1:ndofs2, j = i:ndofs2
            value = 0
            for r in nzrange(cscmat, i)
                for r2 in nzrange(cscmat, j)
                    if rows[r] == rows[r2]
                        value += valsB[r] * valsB[r2] / diagA[rows[r]]
                        break
                    end
                end
            end
            _addnz(S,i,j,value,1)
            if j != i
                _addnz(S,j,i,value,1)
            end
        end

        # compute B' inv(A_diag) f (currently assuming B = C)
        vb.entries ./= diagA
        
        for i = 1:ndofs2
            for r in nzrange(cscmat, i)
                CinvAf[i] += valsB[r] * vb.entries[rows[r]]
            end
        end
        flush!(S)
        O.storage_LHS = S
        O.storage_RHS = CinvAf
        O.storage_ready = true
        O.last_ndofs = ndofs1
    end
    return nothing
end

function assemble!(A::FEMatrixBlock, SC, j::Int, k::Int, o::Int,  O::SchurComplement{T}, CurrentSolution::FEVector; time = 0, At = nothing) where {T}

    # compute Schur complement (if not already done)
    assembleSchurComplement(SC, O, CurrentSolution; time = time)

    # add LHS Schur complement to A
    @logmsg DeepInfo "Applying $(O.name)"
    addblock!(A,O.storage_LHS; factor = O.factor)
    return nothing
end


function assemble!(b::FEVectorBlock, SC, j::Int, o::Int, O::SchurComplement{T}, CurrentSolution::FEVector; factor = 1, time = 0) where {T}

    # compute Schur complement (if not already done)
    assembleSchurComplement(SC, O, CurrentSolution; time = time)
    
    # add RHS Schur complement to b
    @logmsg DeepInfo "Applying $(O.name)"
    addblock!(b,O.storage_RHS; factor = O.factor)
    return nothing
end

# LHS operator is assembled to RHS block (due to subiteration configuration)
function assemble!(b::FEVectorBlock, SC, j::Int, k::Int, o::Int, O::SchurComplement, CurrentSolution::FEVector; factor = 1, time::Real = 0, fixed_component = 0)
    @warn "unknown with Schur complement in its equation should be part of solve, doing nothing here" # how to handle this case approriately ?
end