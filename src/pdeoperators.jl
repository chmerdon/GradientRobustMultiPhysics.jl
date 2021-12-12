
# type to steer when a PDE block is (re)assembled
abstract type AbstractAssemblyTrigger end
abstract type AssemblyAuto <: AbstractAssemblyTrigger end # triggers automatic decision when block is reassembled at solver configuration level
abstract type AssemblyFinal <: AbstractAssemblyTrigger end   # is only assembled after solving
abstract type AssemblyAlways <: AbstractAssemblyTrigger end     # is always (re)assembled
    abstract type AssemblyEachTimeStep <: AssemblyAlways end     # is (re)assembled in each timestep
        abstract type AssemblyInitial <: AssemblyEachTimeStep end    # is only assembled in initial assembly
            abstract type AssemblyNever <: AssemblyInitial end   # is never assembled



#######################
# AbstractPDEOperator #
#######################
#
# to describe operators in the (weak form of the) PDE
#
# some intermediate layer that knows nothing of the FE discretisatons
# but know about their connectivities to help devide fixpoint iterations
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
    action::AbstractAction
    action_rhs::AbstractAction
    action_eval::AbstractAction
    apply_action_to::Array{Int,1}
    fixed_arguments::Array{Int,1}
    fixed_arguments_ids::Array{Int,1}
    newton_arguments::Array{Int,1}
    factor::T
    regions::Array{Int,1}
    transposed_assembly::Bool
    transposed_copy::Bool
    store_operator::Bool
    assembly_trigger::Type{<:AbstractAssemblyTrigger}
    storage::Union{AbstractVector{T},AbstractMatrix{T}}
    PDEOperator{T,APT,AT}(name,ops) where {T <: Real, APT <: AssemblyPatternType, AT <: AssemblyType} = new{T,APT,AT}(name,ops,NoAction(),NoAction(),NoAction(),[1],[],[],[],1,[0],false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,factor) where {T,APT,AT} = new{T,APT,AT}(name,ops,NoAction(),NoAction(),NoAction(),[1],[],[],[],factor,[0],false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,action,apply_to,[],[],[],factor,[0],false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,action,apply_to,[],[],[],factor,regions,false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions,store) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,action,apply_to,[],[],[],factor,regions,false,false,store,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions,store,assembly_trigger) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,action,apply_to,[],[],[],factor,regions,false,false,store,assembly_trigger)
end 

## provisoric copy method for PDEoperators (needed during assignment of nonlinear operators, where each copy is related to partial derivatives of each related unknown)
export copy
function Base.copy(O::PDEOperator{T,APT,AT}) where{T,APT,AT}
    cO = PDEOperator{T,APT,AT}(O.name,copy(O.operators4arguments))
    cO.action = O.action # deepcopy would lead to errors !!!
    cO.action_rhs = O.action_rhs # deepcopy would lead to errors !!!
    cO.action_eval = O.action_eval # deepcopy would lead to errors !!! 
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
    timedependent = typeof(O.action) <: NoAction ? false : is_timedependent(O.action.kernel)
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

restricts a PDEOperator by fixing arguments with the arguments of the full PDEDescription that match the specified argument ids.
"""
function restrict_operator(O::PDEOperator; fixed_arguments = [], fixed_arguments_ids = [], factor = 1)
    Or = deepcopy(O)
    Or.store_operator = false
    Or.factor *= factor
    APT = typeof(O).parameters[2]
    if APT <: APT_BilinearForm
        Or.fixed_arguments = fixed_arguments
        Or.fixed_arguments_ids = fixed_arguments_ids
    elseif APT <: APT_TrilinearForm
        Or.fixed_arguments = fixed_arguments
        Or.fixed_arguments_ids = fixed_arguments_ids
    else
        @error "restriction of this operator is not possible"
    end
    return Or
end


"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = κ (∇u,∇v) where kappa is some constant (diffusion) coefficient.
"""
function LaplaceOperator(κ = 1.0; name = "auto", AT::Type{<:AssemblyType} = ON_CELLS, ∇ = Gradient, regions::Array{Int,1} = [0], store::Bool = false)
    if name == "auto"
        name = "(∇u,∇v)"
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

constructor for a bilinearform that describes a(u,v) = (αu,v) or (u,αv) with some coefficient α that can be a number or an AbstractDataFunction.
"""
function ReactionOperator(α = 1.0, ncomponents = 1; name = "auto", AT::Type{<:AssemblyType} = ON_CELLS, id = Identity, regions::Array{Int,1} = [0], store::Bool = false)
    if name == "auto"
        name = "(u,v)"
        if typeof(α) <: Real
            if α != 1.0
                name = "$α " * name
            end
        elseif typeof(α) <: UserData{AbstractDataFunction}
            name = "(αu,v)"
        end
    end
    if typeof(α) <: Real
        return PDEOperator{Float64, APT_SymmetricBilinearForm, AT}(name,[id,id], NoAction(), [1], α, regions, store, AssemblyInitial)
    elseif typeof(α) <: UserData{AbstractDataFunction}
        xdim = α.dimensions[1]
        function reaction_function_func_xt()
            eval_coeff = zeros(Float64,ncomponents)
            function closure(result, input, x, t)
                # evaluate beta
                eval_data!(eval_coeff,α,x,t)
                # compute alpha*u
                for j = 1 : ncomponents
                    result[j] = eval_coeff[j] * input[j]
                end
            end    
        end    
        action_kernel = ActionKernel(reaction_function_func_xt(), [ncomponents, xdim]; dependencies = "XT", quadorder = α.quadorder)
        return PDEOperator{Float64, APT_BilinearForm, AT}(name, [id, id], Action{Float64}( action_kernel), [1], 1, regions, store, AssemblyAuto)
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
        name = "($operator(v),q)"
    end
    O = PDEOperator{Float64, APT_BilinearForm, AT}(name,[operator, Identity], action, [1], factor, regions, store, AssemblyInitial)
    O.transposed_copy = true
    return O
end


"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (μ ∇u,∇v) where C is the 1D stiffness tensor for given μ.
    
"""
function HookStiffnessOperator1D(μ; name = "(μ ∇u,∇v)", regions::Array{Int,1} = [0], ∇ = TangentialGradient, store::Bool = false)
    return PDEOperator{Float64, APT_BilinearForm, ON_CELLS}(name,[∇, ∇], NoAction(), [1], μ, regions, store, AssemblyInitial)
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (C ϵ(u), ϵ(v)) where C is the 2D stiffness tensor
for isotropic media in Voigt notation, i.e.
C ϵ(u) = 2 μ ϵ(u) + λ tr(ϵ(u)) for Lame parameters μ and λ
    
    In Voigt notation C is a 3 x 3 matrix
    C = [c11,c12,  0
         c12,c11,  0
           0,  0,c33]
    
    where c33 = μ, c12 = λ and c11 = 2*c33 + c12

Note: ϵ is the symmetric part of the gradient (in Voigt notation)
    
"""
function HookStiffnessOperator2D(μ, λ; 
    name = "(C(μ,λ) ϵ(u),ϵ(v))", 
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
    action_kernel = ActionKernel(tensor_apply_2d, [3,3]; dependencies = "", quadorder = 0)
    action = Action{Float64}( action_kernel)
    return PDEOperator{Float64, APT_BilinearForm, AT}(name,[ϵ, ϵ], action, [1], 1, regions, store, AssemblyInitial)
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (C ϵ(u), ϵ(v)) where C is the 3D stiffness tensor
for isotropic media in Voigt notation, i.e.
C ϵ(u) = 2 μ ϵ(u) + λ tr(ϵ(u)) for Lame parameters μ and λ

    In Voigt notation C is a 6 x 6 matrix
    C = [c11,c12,c12,  0,  0,  0
         c12,c11,c12,  0,  0,  0
         c12,c12,c11,  0,  0,  0
           0,  0,  0,c44,  0,  0
           0,  0,  0,  0,c44,  0
           0,  0,  0,  0,  0,c44]   

    where c44 = μ, c12 = λ and c11 = 2*c44 + c12

Note: ϵ is the symmetric part of the gradient (in Voigt notation)
    
"""
function HookStiffnessOperator3D(μ, λ;
    name = "(C(μ,λ) ϵ(u),ϵ(v))", 
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
    action_kernel = ActionKernel(tensor_apply_3d, [6,6]; dependencies = "", quadorder = 0)
    action = Action{Float64}( action_kernel)
    return PDEOperator{Float64, APT_BilinearForm, AT}(name,[ϵ, ϵ], action, [1], 1, regions, store, AssemblyInitial)
end




"""
````
function BilinearForm(
    operators::Array{AbstractFunctionOperator,1},
    action::AbstractAction = NoAction();
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    APT::Type{<:APT_BilinearForm} = APT_BilinearForm,
    apply_action_to = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false,
    store::Bool = false)
````

abstract bilinearform constructor that assembles
- b(u,v) = int_regions action(operator1(u)) * operator2(v) if apply_action_to = 1
- b(u,v) = int_regions operator1(u) * action(operator2(v)) if apply_action_to = 2

The optional arguments AT and regions specifies on which grid item the operator lives/assembles, while store toggles the separate storage for the operator
(which is advisable if it is not alone i an otherweise nonlinear block of a PDEDescription). With the optional argument APT one can trigger different subpatterns
like APT_SymmetricBilinearForm (assembles only a triangular block) or APT_LumpedBilinearForm (assembles only the diagonal).
"""
function BilinearForm(
    operators::Array{DataType,1},
    action::AbstractAction = NoAction();
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    APT::Type{<:APT_BilinearForm} = APT_BilinearForm,
    apply_action_to = [1],
    factor = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false,
    store::Bool = false)

    # check formalities
    @assert apply_action_to in [[1],[2]] "Action must be applied to [1] or [2]"

    # construct PDEoperator
    if name == "auto"
        name = apply_action_to == 1 ? "A($(operators[1])(u)):$(operators[2])(v)" : "$(operators[1])(u):A($(operators[2])(v))"
    end
    
    O = PDEOperator{Float64, APT, AT}(name,operators, action, apply_action_to, factor, regions, store, AssemblyAuto)
    O.transposed_assembly = transposed_assembly
    return O
end


"""
````
function TrilinearForm(
    operators::Array{AbstractFunctionOperator,1},
    a_from::Int,
    a_to::Int,
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)
````

abstract trilinearform constructor that assembles
- c(a,u,v) = (action(operators[1](a),operators[2](u)), operators[3](v))

where u and are the ansatz and test function coressponding to the PDE coordinates and a is an additional unknown of the PDE.
The argument a can be moved to the other positions with a_to and gets it data from unknown a_from of the full PDEdescription.

The optional arguments AT and regions specifies on which grid item the operator lives/assembles,

Also note that this operator is always marked as nonlinear by the Solver configuration.
"""
function TrilinearForm(
    operators::Array{DataType,1},
    a_from::Int,
    a_to::Int,
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    factor = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)

    if name == "auto"
        if a_to == 1
            name = "(A($(operators[1])(a),$(operators[2])(u)), $(operators[3])(v))"
        elseif a_to == 2
            name = "(A($(operators[1])(u),$(operators[2])(a)), $(operators[3])(v))"
        elseif a_to == 3
            name = "(A($(operators[1])(u),$(operators[2])(v)), $(operators[3])(a))"
        end
    end
        
    O = PDEOperator{Float64, APT_TrilinearForm, AT}(name,operators, action, [1,2], factor, regions)
    O.fixed_arguments = [a_to]
    O.fixed_arguments_ids = [a_from]
    O.transposed_assembly = transposed_assembly
    return O
end

"""
````
function ConvectionOperator(
    a_from::Int, 
    beta_operator::Type{<:AbstractFunctionOperator},
    xdim::Int,
    ncomponents::Int;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    fixed_argument::Int = 1,
    factor = 1,
    ansatzfunction_operator::Type{<:AbstractFunctionOperator} = Gradient,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0],
    auto_newton::Bool = false,
    quadorder = 0)
````

constructs a trilinearform for a convection term of the form c(a,u,v) = (beta_operator(a)*ansatzfunction_operator(u),testfunction_operator(v))
where a_from is the id of some unknown of the PDEDescription; xdim is the space dimension (= number of components of beta_operato(a)) and ncomponents is the number of components of u.
With fixed_argument = 2 a and u can switch their places, i.e.  c(u,a,v) = (beta_operator(u)*grad(a),v),
With auto_newton = true a Newton scheme for c(u,u,v) is automatically derived (and fixed_argument is ignored).

"""
function ConvectionOperator(
    a_from::Int, 
    beta_operator::Type{<:AbstractFunctionOperator},
    xdim::Int,
    ncomponents::Int;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    fixed_argument::Int = 1,
    factor = 1,
    ansatzfunction_operator::Type{<:AbstractFunctionOperator} = Gradient,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0],
    auto_newton::Bool = false,
    transposed_assembly::Bool = true,
    quadorder = 0)

    # action input consists of two inputs
    # input[1:xdim] = operator1(a)
    # input[xdim+1:end] = grad(u)
    function convection_function_fe(result::Array{<:Real,1}, input::Array{<:Real,1})
        for j = 1 : ncomponents
            result[j] = 0
            for k = 1 : xdim
                result[j] += input[k]*input[xdim+(j-1)*xdim+k]
            end
        end
    end    
    argsizes = [ncomponents, xdim + ncomponents*xdim]
    if auto_newton
        ## generates a nonlinear form with automatic Newton operators by AD
        if name == "auto"
            name = "(($beta_operator(u) ⋅ $ansatzfunction_operator) u, $(testfunction_operator)(v))"
        end
        return NonlinearForm([beta_operator, ansatzfunction_operator], [a_from,a_from], testfunction_operator, convection_function_fe,argsizes; name = name, ADnewton = true, quadorder = quadorder)     
    else
        ## returns linearised convection operators as a trilinear form (Picard iteration)
        action_kernel = ActionKernel(convection_function_fe,argsizes; dependencies = "", quadorder = quadorder)
        convection_action = Action{Float64}( action_kernel)
        a_to = fixed_argument
        if name == "auto"
            if a_to == 1
                name = "(($beta_operator(a) ⋅ $ansatzfunction_operator) u, $(testfunction_operator)(v))"
            elseif a_to == 2
                name = "(($beta_operator(u) ⋅ $ansatzfunction_operator) a, $(testfunction_operator)(v))"
            elseif a_to == 3
                name = "(($beta_operator(u) ⋅ $ansatzfunction_operator) v, $(testfunction_operator)(a))"
            end
        end
        
        O = PDEOperator{Float64, APT_TrilinearForm, AT}(name,[beta_operator,ansatzfunction_operator,testfunction_operator], convection_action, [1,2], factor, regions)
        O.fixed_arguments = [a_to]
        O.fixed_arguments_ids = [a_from]
        O.transposed_assembly = transposed_assembly
        return O
    end
end

"""
$(TYPEDSIGNATURES)

constructor for a trilinearform that describes a(u,v) = (beta x curl(u),v)
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
    ansatzfunction_operator::Type{<:AbstractFunctionOperator} = Curl2D,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
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
        action_kernel = ActionKernel(rotationform_2d(),[2, 3]; dependencies = "", quadorder = 0)
        convection_action = Action{Float64}( action_kernel)
        if name == "auto"
            name = "((β × ∇) u, v)"
        end
        O = PDEOperator{Float64, APT_TrilinearForm, AT}(name,[beta_operator,ansatzfunction_operator,testfunction_operator], convection_action, [1,2], factor, regions)
        O.fixed_arguments = [1]
        O.fixed_arguments_ids = [beta]
        O.transposed_assembly = true
        return O
    else
        @error "The rotation form of the convection operator is currently only available in 2D (in 3D please implement it yourself using TrilinearForm and a user-defined action)"
    end
end





"""
````
function NonlinearForm(
    operator1::Array{DataType,1},
    coeff_from::Array{Int,1},
    operator2::Type{<:AbstractFunctionOperator},
    action_kernel::Function,
    argsizes::Array{Int,1},
    dim::Int;
    name::String = "nonlinear form",
    AT::Type{<:AssemblyType} = ON_CELLS,
    ADnewton::Bool = false,
    action_kernel_rhs = nothing,
    factor = 1,
    regions = [0])
````

generates an abstract nonlinearform operator G. 
The array coeff_from stores the ids of the unknowns that should be used to evaluate the operators. The array argsizes is a vector with two entries where the first one
is the length of the expected result vector and the second one is the length of the input vector.

If ADnewton == true, the specified action_kernel is automatically differentiated to assemble the Jacobian DG
and setup a Newton iteration. The action_kernel has to be a function of the interface 

    function name(result,input)

where input is a vector of the operators of the solution and result is what then is multiplied with operator2 of the testfunction.
Given some operator G(u), the Newton iteration reads DG u_next = DG u - G(u) which is added to the rest of the (linear) operators in the PDEDescription.


If ADnewton == false, the user is epected to prescribe a linearisation of the nonlinear operator. In this case the action_kernel has to satisfy the interface

    function name(result, input_current, input_ansatz)

where input_current is a vector of the operators of the solution and input_ansatz is a vecor with the operators evaluated at one of the basis functions.
If necessary, also a right-hand side action in the same format can be prescribed in action_kernel_rhs. 

Note 1: The AD feature matured a bit, but still is to be considered experimental.

Note 2: The limitation that the nonlinearity only can depend on one unknown of the PDE was recently lifted, however the behavior how to assign this operator to the PDE
may be revised in future. Currently, the nonlinearity can indeed depend on arbitrary unknowns (i.e. coeff_from may contain more than one different unknown ids),
which will lead to copies of the operator assigned also to off-diagonal blocks which are then related to partial derivatives with respect to the other unknowns
(i.e. input_ansatz will only contain the operator evaluations that coresspond to the unknown of the subblock it is evaluated at, all other entries are zero).
The subblock assignment of the copies is done automatically by the add_operator! function.

"""
function NonlinearForm(
    operator1::Array{DataType,1},
    coeff_from::Array{Int,1},
    operator2::Type{<:AbstractFunctionOperator},
    action_kernel, # should be a function of input (result, input) or matching the specified dependencies
    argsizes::Array{Int,1};
    name::String = "nonlinear form",
    AT::Type{<:AssemblyType} = ON_CELLS,
    ADnewton::Bool = false,
    action_kernel_rhs = nothing,
    dependencies = "",
    quadorder::Int = 0,
    factor = 1,
    regions = [0])

    ### Newton scheme for a nonlinear operator G(u) is
    ## seek u_next such that : DG u_next = DG u - G(u)

    if length(argsizes) == 2
        push!(argsizes,argsizes[2])
    end

    if ADnewton
        name = name * " [AD-Newton]"
        # the action for the derivative matrix DG is calculated by automatic differentiation (AD)
        # from the given action_kernel of the operator G

        # for differentation other dependencies of the action_kernel are fixed
        result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
        input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[3])
        #jac_temp::Matrix{Float64} = Matrix{Float64}(undef,argsizes[1],argsizes[3])
        #Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        Dresult = DiffResults.JacobianResult(result_temp,input_temp)
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        temp::Array{Float64,1} = zeros(Float64, argsizes[1])
        if dependencies == "X"
            reduced_action_kernel_x(x) = (result,input) -> action_kernel(result,input,x)
            cfg =ForwardDiff.JacobianConfig(reduced_action_kernel_x([1.0,1.0,1.0]), result_temp, input_temp, ForwardDiff.Chunk{argsizes[3]}())
            function newton_kernel_x(result::Array{<:Real,1}, input_current::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, x)
                ForwardDiff.vector_mode_jacobian!(Dresult, reduced_action_kernel_x(x), result, input_current, cfg)
                jac = DiffResults.jacobian(Dresult)
                for j = 1 : argsizes[1]
                    result[j] = 0
                    for k = 1 : argsizes[2]
                        result[j] += jac[j,k] * input_ansatz[k]
                    end
                end
                return nothing
            end

            # the action for the RHS just evaluates DG and G at input_current
            function rhs_kernel_x(result::Array{<:Real,1}, input_current::Array{<:Real,1}, x)
                fill!(result,0)
                newton_kernel_x(result, input_current, input_current, x)
                reduced_action_kernel_x(x)(temp, input_current)
                for j = 1 : argsizes[1]
                    result[j] -= temp[j]
                end
                return nothing
            end
            newton_action_kernel = NLActionKernel(newton_kernel_x, argsizes; dependencies = dependencies, quadorder = quadorder)
            action = Action{Float64}( newton_action_kernel)
            rhs_action_kernel = ActionKernel(rhs_kernel_x, argsizes; dependencies = dependencies, quadorder = quadorder)
            action_rhs = Action{Float64}( rhs_action_kernel)
        elseif dependencies == "T"
            reduced_action_kernel_t(t) = (result,input) -> action_kernel(result,input,t)
            cfg =ForwardDiff.JacobianConfig(reduced_action_kernel_t(0.0), result_temp, input_temp, ForwardDiff.Chunk{argsizes[3]}())
            function newton_kernel_t(result::Array{<:Real,1}, input_current::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, t)
                ForwardDiff.vector_mode_jacobian!(Dresult, reduced_action_kernel_t(t), result, input_current, cfg)
                jac = DiffResults.jacobian(Dresult)
                for j = 1 : argsizes[1]
                    result[j] = 0
                    for k = 1 : argsizes[2]
                        result[j] += jac[j,k] * input_ansatz[k]
                    end
                end
                return nothing
            end

            # the action for the RHS just evaluates DG and G at input_current
            function rhs_kernel_t(result::Array{<:Real,1}, input_current::Array{<:Real,1}, t)
                fill!(result,0)
                newton_kernel_t(result, input_current, input_current, t)
                reduced_action_kernel_t(t)(temp, input_current)
                for j = 1 : argsizes[1]
                    result[j] -= temp[j]
                end
                return nothing
            end
            newton_action_kernel = NLActionKernel(newton_kernel_t, argsizes; dependencies = dependencies, quadorder = quadorder)
            action = Action{Float64}( newton_action_kernel)
            rhs_action_kernel = ActionKernel(rhs_kernel_t, argsizes; dependencies = dependencies, quadorder = quadorder)
            action_rhs = Action{Float64}( rhs_action_kernel)
        elseif dependencies == "XT"
            reduced_action_kernel_xt(x,t) = (result,input) -> action_kernel(result,input,x,t)
            cfg =ForwardDiff.JacobianConfig(reduced_action_kernel_xt([1.0,1.0,1.0],0.0), result_temp, input_temp, ForwardDiff.Chunk{argsizes[3]}())
            function newton_kernel_xt(result::Array{<:Real,1}, input_current::Array{<:Real,1}, input_ansatz::Array{<:Real,1}, x, t)
                ForwardDiff.vector_mode_jacobian!(Dresult, reduced_action_kernel_xt(x,t), result, input_current, cfg)
                jac = DiffResults.jacobian(Dresult)
                for j = 1 : argsizes[1]
                    result[j] = 0
                    for k = 1 : argsizes[2]
                        result[j] += jac[j,k] * input_ansatz[k]
                    end
                end
                return nothing
            end

            # the action for the RHS just evaluates DG and G at input_current
            function rhs_kernel_xt(result::Array{<:Real,1}, input_current::Array{<:Real,1}, x, t)
                fill!(result,0)
                newton_kernel_xt(result, input_current, input_current, x, t)
                reduced_action_kernel_xt(x, t)(temp, input_current)
                for j = 1 : argsizes[1]
                    result[j] -= temp[j]
                end
                return nothing
            end
            newton_action_kernel = NLActionKernel(newton_kernel_xt, argsizes; dependencies = dependencies, quadorder = quadorder)
            action = Action{Float64}( newton_action_kernel)
            rhs_action_kernel = ActionKernel(rhs_kernel_xt, argsizes; dependencies = dependencies, quadorder = quadorder)
            action_rhs = Action{Float64}( rhs_action_kernel)
        elseif dependencies == ""
            cfg = ForwardDiff.JacobianConfig(action_kernel, result_temp, input_temp, ForwardDiff.Chunk{argsizes[3]}())
            function newton_kernel(result, input_current, input_ansatz)
                Dresult = ForwardDiff.chunk_mode_jacobian!(Dresult, action_kernel, result, input_current, cfg)
                jac = DiffResults.jacobian(Dresult)
                for j = 1 : argsizes[1]
                    result[j] = 0
                    for k = 1 : argsizes[2]
                        result[j] += jac[j,k] * input_ansatz[k]
                    end
                end
                return nothing
            end

            # the action for the RHS just evaluates DG and G at input_current
            function rhs_kernel(result, input_current)
                fill!(result,0)
                newton_kernel(result, input_current, input_current)
                action_kernel(temp, input_current)
                for j = 1 : argsizes[1]
                    result[j] -= temp[j]
                end
                return nothing
            end
            newton_action_kernel = NLActionKernel(newton_kernel, argsizes; dependencies = dependencies, quadorder = quadorder)
            action = Action{Float64}( newton_action_kernel)
            rhs_action_kernel = ActionKernel(rhs_kernel, argsizes; dependencies = dependencies, quadorder = quadorder)
            action_rhs = Action{Float64}( rhs_action_kernel)
        else
            @error "Currently nonlinear kernels may only depend on the additional argument(s) x or t"
        end
    else

        # take action_kernel as nonlinear action_kernel
        # = user specifies linearisation of nonlinear operator
        nlform_action_kernel = NLActionKernel(action_kernel, argsizes; dependencies = dependencies, quadorder = quadorder)

        action = Action{Float64}( nlform_action_kernel)
        if action_kernel_rhs != nothing
            action_rhs = Action{Float64}( action_kernel_rhs)
        else
            action_rhs = nothing
        end
    end

    append!(operator1, [operator2])
    O = PDEOperator{Float64, APT_NonlinearForm, AT}(name, operator1, action, 1:(length(coeff_from)), 1, regions)
    O.fixed_arguments = 1:length(coeff_from)
    O.fixed_arguments_ids = coeff_from
    O.newton_arguments = 1 : length(coeff_from) # if depended on different ids, operator is later splitted into several operators where newton_arguments refer only to subset
    O.factor = factor
    O.action_rhs = action_rhs === nothing ? NoAction() : action_rhs
    O.transposed_assembly = true

    # eval action
    O.action_eval = Action{Float64}( action_kernel, argsizes; dependencies = dependencies, quadorder = quadorder)
    return O
end


"""
$(TYPEDSIGNATURES)

generates a linearform from an action (whose input dimension has to equal the result dimension of operator and the result dimension has to be 1), L(v) = (A(operator(v)),1)
    
"""
function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    factor = 1,
    store::Bool = false)

    @assert action.argsizes[1] == 1 "Actions for right-hand sides must have result dimension 1!"

    if name == "auto"
        name = "(A($operator(v)), 1)"
    end
    O = PDEOperator{Float64, APT_LinearForm, AT}(name, [operator], action, [1], 1, regions, store, AssemblyAuto)
    O.factor = factor
    return O 
end


"""
$(TYPEDSIGNATURES)

generates a linearform from a given UserData{<:DataFunction} f (that should have the same result dimensions as the length of result of the operator applied to the testfunction),
i.e. L(v) = (f,operator(v))
    
"""
function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    regions::Array{Int,1},
    f::UserData{<:AbstractDataFunction};
    name = "auto",
    AT::Type{<:AssemblyType} = ON_CELLS,
    factor = 1,
    store::Bool = false)

    if name == "auto"
        name = "($(f.name), $operator(v))"
    end

    action = fdot_action(Float64,f)
    O = PDEOperator{Float64, APT_LinearForm, AT}(name,[operator], action, [1], 1, regions, store, AssemblyAuto)
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
    β::UserData{AbstractDataFunction},
    ncomponents::Int; 
    name = "auto", 
    store::Bool = false,
    AT::Type{<:AssemblyType} = ON_CELLS,
    ansatzfunction_operator::Type{<:AbstractFunctionOperator} = Gradient,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    transposed_assembly::Bool = true,
    regions::Array{Int,1} = [0])

    T = Float64
    xdim = β.dimensions[1]
    if is_timedependent(β) && is_xdependent(β)
        function convection_function_func_xt() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input, x, time)
                # evaluate β
                eval_data!(convection_vector,β,x,time)
                # compute (β ⋅ ∇) u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func_xt(), [ncomponents, ncomponents*xdim]; dependencies = "XT", quadorder = β.quadorder)
    elseif !is_timedependent(β) && !is_xdependent(β)
        function convection_function_func() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input)
                # evaluate β
                eval_data!(convection_vector,β, nothing, nothing)
                # compute (β ⋅ ∇) u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func(), [ncomponents, ncomponents*xdim]; dependencies = "", quadorder = β.quadorder)
    elseif !is_timedependent(β) && is_xdependent(β)
        function convection_function_func_x() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input, x)
                # evaluate β
                eval_data!(convection_vector,β,x,nothing)
                # compute (β ⋅ ∇) u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func_x(), [ncomponents, ncomponents*xdim]; dependencies = "X", quadorder = β.quadorder)
    elseif is_timedependent(β) && !is_xdependent(β)
        function convection_function_func_t() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input, t)
                # evaluate β
                eval_data!(convection_vector,β,nothing,t)
                # compute (β ⋅ ∇) u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func_t(), [ncomponents, ncomponents*xdim]; dependencies = "T", quadorder = β.quadorder)
    end
    if name == "auto"
        name = "((β ⋅ $(ansatzfunction_operator)) u, $testfunction_operator(v))"
    end

    O = PDEOperator{T, APT_BilinearForm, AT}(name, [ansatzfunction_operator, testfunction_operator], Action{T}(action_kernel), [1], 1, regions, store, AssemblyAuto)
    O.transposed_assembly = transposed_assembly
    return O
end



function update_storage!(O::PDEOperator, CurrentSolution::FEVector{T,Tv,Ti}, j::Int, k::Int; factor = 1, time::Real = 0) where {T,Tv,Ti}
    @logmsg MoreInfo "Updating storage of PDEOperator $(O.name) in LHS block [$j,$k]"

    set_time!(O.action, time)
    APT = typeof(O).parameters[2]
    AT = typeof(O).parameters[3]
    if APT <: APT_BilinearForm
        FES = Array{FESpace{Tv,Ti},1}(undef, 2)
        FES[1] = CurrentSolution[j].FES
        FES[2] = CurrentSolution[k].FES
    else
        @error "No storage functionality available for this operator!"
    end
    O.storage = ExtendableSparseMatrix{T,Int64}(FES[1].ndofs,FES[2].ndofs)
    Pattern = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
    assemble!(O.storage, Pattern; transposed_assembly = O.transposed_assembly, factor = factor, skip_preps = false)
    flush!(O.storage)
end

function update_storage!(O::PDEOperator, CurrentSolution::FEVector{T,Tv,Ti}, j::Int; factor = 1, time::Real = 0) where {T,Tv,Ti}

    @logmsg MoreInfo "Updating storage of PDEOperator $(O.name) in RHS block [$j]"

    set_time!(O.action, time)
    APT = typeof(O).parameters[2]
    AT = typeof(O).parameters[3]
    if APT <: APT_LinearForm
        FES = Array{FESpace{Tv,Ti},1}(undef, 1)
        FES[1] = CurrentSolution[j].FES
    else
        @error "No storage functionality available for this operator!"
    end
    O.storage = zeros(T,FES[1].ndofs)
    Pattern = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action_rhs,O.apply_action_to,O.regions)
    assemble!(O.storage, Pattern; factor = factor, skip_preps = false)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock{TvM,TiM,TvG,TiG}, CurrentSolution) where{T,TvM,TiM,TvG,TiG,APT<:APT_BilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, 2)
    FES[1] = O.transposed_assembly ? A.FESY : A.FESX
    FES[2] = O.transposed_assembly ? A.FESX : A.FESY
    return AssemblyPattern{APT, TvM, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock{TvV,TvG,TiG}, CurrentSolution::FEVector; non_fixed::Int = 1, fixed_id = 1) where{T,TvV,TvG,TiG,APT<:APT_BilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, 2)
    if length(O.fixed_arguments) == 1
        # assume it is a restricted bilinearform
        non_fixed = O.fixed_arguments[1] == 1 ? 2 : 1
        fixed_id = O.fixed_arguments_ids[1]
        FES[non_fixed] = b.FES
        FES[non_fixed == 1 ? 2 : 1] = CurrentSolution[fixed_id].FES
    else
        # assume it is a LHS bilinearform that is assembled to the RHS with a fixed argument
        FES[non_fixed] = b.FES
        FES[non_fixed == 1 ? 2 : 1] = CurrentSolution[fixed_id].FES
    end
    return AssemblyPattern{APT, TvV, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock{TvV,TvG,TiG}, CurrentSolution; non_fixed::Int = 1, fixed_id = 1) where{T,TvV,TvG,TiG,APT<:APT_LinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, 1)
    FES[1] = b.FES
    return AssemblyPattern{APT, TvV, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock{TvM,TiM,TvG,TiG}, CurrentSolution::FEVector) where{T,TvM,TiM,TvG,TiG,APT<:APT_TrilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, 3)
    FES[O.fixed_arguments[1]] = CurrentSolution[O.fixed_arguments_ids[1]].FES
    if O.fixed_arguments == [1]
        FES[2] = O.transposed_assembly ? A.FESY : A.FESX
        FES[3] = O.transposed_assembly ? A.FESX : A.FESY
    elseif O.fixed_arguments == [2]
        FES[1] = O.transposed_assembly ? A.FESY : A.FESX
        FES[3] = O.transposed_assembly ? A.FESX : A.FESY
    elseif O.fixed_arguments == [3]
        FES[1] = O.transposed_assembly ? A.FESY : A.FESX
        FES[2] = O.transposed_assembly ? A.FESX : A.FESY
    end
    return AssemblyPattern{APT, TvM, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock{TvV,TvG,TiG}, CurrentSolution::FEVector; non_fixed::Int = 1, fixed_id = 1) where{T,TvV,TvG,TiG,APT<:APT_TrilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace{TvG,TiG},1}(undef, 3)
    FES[O.fixed_arguments[1]] = CurrentSolution[O.fixed_arguments_ids[1]].FES
    if length(O.fixed_arguments) == 2
        # a restricted Trilineraform is assembled as a RHS operator
        FES[O.fixed_arguments[2]] = CurrentSolution[O.fixed_arguments_ids[2]].FES
        non_fixed = setdiff([1,2,3], O.fixed_arguments)[1]
        FES[non_fixed] = b.FES
    else # it is assumed that a LHS Trilinearform is assembled into RHS block
        if O.fixed_arguments == [1]
            FES[non_fixed == 1 ? 2 : 3] = O.transposed_assembly ? CurrentSolution[fixed_id].FES : b.FES
            FES[non_fixed == 1 ? 3 : 2] = O.transposed_assembly ? b.FES : CurrentSolution[fixed_id].FES
        elseif O.fixed_arguments == [2]
            FES[non_fixed == 1 ? 1 : 3] = O.transposed_assembly ? CurrentSolution[fixed_id].FES : b.FES
            FES[non_fixed == 1 ? 3 : 1] = O.transposed_assembly ? b.FES : CurrentSolution[fixed_id].FES
        elseif O.fixed_arguments == [3]
            FES[non_fixed == 1 ? 2 : 3] = O.transposed_assembly ? CurrentSolution[fixed_id].FES : b.FES
            FES[non_fixed == 1 ? 3 : 2] = O.transposed_assembly ? b.FES : CurrentSolution[fixed_id].FES
        end
    end
    return AssemblyPattern{APT, TvV, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
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


"""
$(TYPEDSIGNATURES)

assembles the operator O into the given FEMatrixBlock A using FESpaces from A. An FEVector CurrentSolution is only needed if the operator involves fixed arguments, e.g. if O is a TrilinearForm.
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


function assemble_operator!(b::FEVectorBlock, O::PDEOperator, CurrentSolution::Union{Nothing,FEVector} = nothing; Pattern = nothing, skip_preps::Bool = false, factor = 1, time::Real = 0)
    if Pattern === nothing
        Pattern = create_assembly_pattern(O, b, CurrentSolution)
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
        addblock!(A,O.storage; factor = O.factor)
        if At !== nothing
            addblock!(At,O.storage; factor = O.factor, transpose = true)
        end
    else
        ## find assembly pattern
        skip_preps = true
        if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            SC.LHS_AssemblyPatterns[j,k][o] = create_assembly_pattern(O, A, CurrentSolution)
            skip_preps = false
        end
        ## assemble
        assemble_operator!(A, O, CurrentSolution; Pattern = SC.LHS_AssemblyPatterns[j,k][o], time = time, At = At, skip_preps = skip_preps)
    end
end


function assemble!(b::FEVectorBlock, SC, j::Int, o::Int, O::PDEOperator, CurrentSolution::FEVector; factor = 1, time::Real = 0)
    if O.store_operator == true
        @logmsg DeepInfo "Adding PDEOperator $(O.name) from storage"
        addblock!(b, O.storage; factor = factor * O.factor)
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
        addblock_matmul!(b,O.storage,CurrentSolution[fixed_component]; factor = factor * O.factor)
    else
        ## find assembly pattern
        skip_preps = true
        if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
            SC.LHS_AssemblyPatterns[j,k][o] = create_assembly_pattern(O, b, CurrentSolution; non_fixed = (j == fixed_component ? 2 : 1), fixed_id = fixed_component)
            skip_preps = false
        end

        ## find fixed arguments by j,k positioning (additional to fixed arguments of operator)
        if typeof(O).parameters[2] <: APT_BilinearForm
            if fixed_component == j
                fixed_arguments = [1]
            elseif fixed_component == k
                fixed_arguments = [2]
            else
                @error "Something went severely wrong..."
            end
            fixed_arguments_ids = [fixed_component]
        elseif typeof(O).parameters[2] <: APT_TrilinearForm
            fixed_arguments = O.fixed_arguments
            fixed_arguments_ids = O.fixed_arguments_ids
            if fixed_component == j
                push!(fixed_arguments, 1)
            elseif fixed_component == k
                push!(fixed_arguments, 2)
            else
                @error "Something went severely wrong..."
            end
            push!(fixed_arguments_ids, fixed_component)
        end

        ## assemble
        set_time!(O.action, time)
        if length(fixed_arguments_ids) > 0
            assemble!(b, SC.LHS_AssemblyPatterns[j,k][o],CurrentSolution[fixed_arguments_ids]; skip_preps = skip_preps, factor = O.factor*factor, fixed_arguments = fixed_arguments)
        else
            assemble!(b, SC.LHS_AssemblyPatterns[j,k][o]; skip_preps = skip_preps, factor = O.factor*factor)
        end
    end
end



############################################
### EVALUATION OF NONLINEAR PDEOPERATORS ###
############################################

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
        SC.LHS_AssemblyPatterns[j,k][o] = ItemIntegrator(T, ON_FACES, [NormalFlux]; name = "u ⋅ n")
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