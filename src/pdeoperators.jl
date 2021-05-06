
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
mutable struct PDEOperator{T <: Real, APT <: AssemblyPatternType, AT <: AbstractAssemblyType} <: AbstractPDEOperator
    name::String
    operators4arguments::Array{DataType,1}
    action::AbstractAction
    action_rhs::AbstractAction
    apply_action_to::Array{Int,1}
    fixed_arguments::Array{Int,1}
    fixed_arguments_ids::Array{Int,1}
    factor::T
    regions::Array{Int,1}
    transposed_assembly::Bool
    transposed_copy::Bool
    store_operator::Bool
    assembly_trigger::Type{<:AbstractAssemblyTrigger}
    storage::Union{<:AbstractVector,<:AbstractMatrix}
    PDEOperator{T,APT,AT}(name,ops) where {T <: Real, APT <: AssemblyPatternType, AT <: AbstractAssemblyType} = new{T,APT,AT}(name,ops,NoAction(),NoAction(),[1],[],[],1,[0],false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,factor) where {T,APT,AT} = new{T,APT,AT}(name,ops,NoAction(),NoAction(),[1],[],[],factor,[0],false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,apply_to,[],[],factor,[0],false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,apply_to,[],[],factor,regions,false,false,false,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions,store) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,apply_to,[],[],factor,regions,false,false,store,AssemblyAuto)
    PDEOperator{T,APT,AT}(name,ops,action,apply_to,factor,regions,store,assembly_trigger) where {T,APT,AT} = new{T,APT,AT}(name,ops,action,action,apply_to,[],[],factor,regions,false,false,store,assembly_trigger)
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
function restrict_operator(O::PDEOperator; fixed_arguments = [], fixed_arguments_ids = [])
    Or = deepcopy(O)
    Or.store_operator = false
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

constructor for a bilinearform that describes a(u,v) = (kappa * nabla u, nabla v) where kappa is some constant diffusion coefficient
"""
function LaplaceOperator(diffusion = 1.0; name = "auto", AT::Type{<:AbstractAssemblyType} = ON_CELLS, gradient_operator = Gradient, regions::Array{Int,1} = [0], store::Bool = false)
    if name == "auto"
        name = "∇(u):∇(v)"
        if typeof(diffusion) <: Real
            if diffusion != 1
                name = "$diffusion " * name
            end
        end
    end
    if typeof(diffusion) <: Real
        O = PDEOperator{Float64, APT_SymmetricBilinearForm, AT}(name,[gradient_operator, gradient_operator], NoAction(), [1], diffusion, regions, store, AssemblyInitial)
        return O
    else
        @error "No standard Laplace operator definition for this diffusion type available, please define your own action and PDEOperator with it."
    end
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (A(u),v) or (u,A(v)) with some user-specified action A
    
"""
function ReactionOperator(coefficient = 1.0; name = "auto", AT::Type{<:AbstractAssemblyType} = ON_CELLS, identity_operator = Identity, regions::Array{Int,1} = [0], store::Bool = false)
    if name == "auto"
        name = "u ⋅ v"
        if typeof(coefficient) <: Float64
            if coefficient != 1.0
                name = "$coefficient " * name
            end
        end
    end
    if typeof(coefficient) <: Float64
        return PDEOperator{Float64, APT_SymmetricBilinearForm, AT}(name,[identity_operator,identity_operator], NoAction(), [1], coefficient, regions, store, AssemblyInitial)
    else
        @error "No standard reaction operator definition for this coefficient type available, please define your own action and PDEOperator with it."
    end
end


"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (A(operator(u)), id(v)) and assembles a second transposed block at the block of the transposed PDE coordinates. It is intended to use
to render one unknown of the PDE the Lagrange multiplier for another unknown by putting this operator on the coressponding subdiagonal block of the PDE description.

Example: LagrangeMultiplier(Divergence) is used to render the pressure the LagrangeMultiplier for the velocity divergence constraint in the Stokes prototype.
"""
function LagrangeMultiplier(operator::Type{<:AbstractFunctionOperator}; name = "auto", AT::Type{<:AbstractAssemblyType} = ON_CELLS, action::AbstractAction = NoAction(), regions::Array{Int,1} = [0], store::Bool = false, factor = -1)
    if name == "auto"
        name = "$operator(v) ⋅ q"
    end
    O = PDEOperator{Float64, APT_BilinearForm, AT}(name,[operator, Identity], action, [1], factor, regions, store, AssemblyInitial)
    O.transposed_copy = true
    return O
end


"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (C grad(u), grad(v)) where C is the 1D stiffness tensor
C grad(u) = mu grad(u)
    
"""
function HookStiffnessOperator1D(mu; name = "C∇u⋅∇v", regions::Array{Int,1} = [0], gradient_operator = TangentialGradient, store::Bool = false)
    function tensor_apply_1d(result, input)
        # just Hook law like a spring where mu is the elasticity modulus
        result[1] = mu*input[1]
        return nothing
    end   
    action_kernel = ActionKernel(tensor_apply_1d, [1,1]; dependencies = "", quadorder = 0)
    action = Action(Float64, action_kernel)
    return PDEOperator{Float64, APT_BilinearForm, ON_CELLS}(name,[gradient_operator, gradient_operator], action, [1], 1, regions, store, AssemblyInitial)
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (C eps(u), eps(v)) where C is the 3D stiffness tensor
for isotropic media in Voigt notation, i.e.
C eps(u) = 2 mu eps(u) + lambda tr(eps(u)) for Lame parameters mu and lambda
    
    In Voigt notation C is a 3 x 3 matrix
    C = [c11,c12,  0
         c12,c11,  0
           0,  0,c33]
    
    where c33 = shear_modulus, c12 = lambda and c11 = 2*c33 + c12
    
"""
function HookStiffnessOperator2D(mu, lambda; 
    name = "Cϵ(u):ϵ(v)", 
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0], 
    gradient_operator = SymmetricGradient, 
    store::Bool = false)

    function tensor_apply_2d(result, input)
        result[1] = (lambda + 2*mu)*input[1] + lambda*input[2]
        result[2] = (lambda + 2*mu)*input[2] + lambda*input[1]
        result[3] = mu*input[3]
        return nothing
    end   
    action_kernel = ActionKernel(tensor_apply_2d, [3,3]; dependencies = "", quadorder = 0)
    action = Action(Float64, action_kernel)
    return PDEOperator{Float64, APT_BilinearForm, AT}(name,[gradient_operator, gradient_operator], action, [1], 1, regions, store, AssemblyInitial)
end

"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (C eps(u), eps(v)) where C is the 3D stiffness tensor
for isotropic media in Voigt notation, i.e. C eps(u) = 2 mu eps(u) + lambda tr(eps(u)) for Lame parameters mu and lambda

    In Voigt notation C is a 6 x 6 matrix
    C = [c11,c12,c12,  0,  0,  0
         c12,c11,c12,  0,  0,  0
         c12,c12,c11,  0,  0,  0
           0,  0,  0,c44,  0,  0
           0,  0,  0,  0,c44,  0
           0,  0,  0,  0,  0,c44]   

    where c44 = shear_modulus, c12 = lambda and c11 = 2*c44 + c12
    
"""
function HookStiffnessOperator3D(mu, lambda;
    name = "Cϵ(u):ϵ(v)", 
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    gradient_operator = SymmetricGradient,
    store::Bool = false)

    function tensor_apply_3d(result, input)
        result[1] = (lambda + 2*mu)*input[1] + lambda*(input[2] + input[3])
        result[2] = (lambda + 2*mu)*input[2] + lambda*(input[1] + input[3])
        result[3] = (lambda + 2*mu)*input[3] + lambda*(input[1] + input[2])
        result[4] = mu*input[4]
        result[5] = mu*input[5]
        result[6] = mu*input[6]
        return nothing
    end   
    action_kernel = ActionKernel(tensor_apply_3d, [6,6]; dependencies = "", quadorder = 0)
    action = Action(Float64, action_kernel)
    return PDEOperator{Float64, APT_BilinearForm, AT}(name,[gradient_operator, gradient_operator], action, [1], 1, regions, store, AssemblyInitial)
end




"""
````
function AbstractBilinearForm(
    operators::Array{AbstractFunctionOperator,1},
    action::AbstractAction = NoAction();
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    apply_action_to = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false,
    store::Bool = false)
````

abstract bilinearform constructor that assembles
- b(u,v) = int_regions action(operator1(u)) * operator2(v) if apply_action_to = 1
- b(u,v) = int_regions operator1(u) * action(operator2(v)) if apply_action_to = 2

The optional arguments AT and regions specifies on which grid item the operator lives/assembles, while store toggles the separate storage for the operator
(which is advisable if it is not alone i an otherweise nonlinear block of a PDEDescription). 
"""
function AbstractBilinearForm(
    operators::Array{DataType,1},
    action::AbstractAction = NoAction();
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    apply_action_to = [1],
    factor = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false,
    store::Bool = false)

    # check formalities
    @assert length(operators) == 2 "BilinearForm needs exactly 2 operators!"
    @assert apply_action_to in [[1],[2]] "Action must be applied to [1] or [2]"

    # construct PDEoperator
    if name == "auto"
        name = apply_action_to == 1 ? "A($(operators[1])(u)):$(operators[2])(v)" : "$(operators[1])(u):A($(operators[2])(v))"
    end
    
    O = PDEOperator{Float64, APT_BilinearForm, AT}(name,operators, action, apply_action_to, factor, regions, store, AssemblyAuto)
    O.transposed_assembly = transposed_assembly
    return O
end


"""
````
function AbstractTrilinearForm(
    operators::Array{AbstractFunctionOperator,1},
    a_from::Int,
    a_to::Int,
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)
````

abstract trilinearform constructor that assembles
- c(a,u,v) = int_regions action(operator1(a),operator2(u)) * operator3(v)

where u and are the ansatz and test function coressponding to the PDE coordinates and a is an additional unknown of the PDE.
The argument a can be moved to the other positions with a_to and gets it data from unknown a_from of the full PDEdescription.

The optional arguments AT and regions specifies on which grid item the operator lives/assembles,

Also note that this operator is always marked as nonlinear by the Solver configuration.
"""
function AbstractTrilinearForm(
    operators::Array{DataType,1},
    a_from::Int,
    a_to::Int,
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)

    if name == "auto"
        if a_to == 1
            name = "A($(operators[1])(a),$(operators[2])(u)) ⋅ $(operators[3])(v)"
        elseif a_to == 2
            name = "A($(operators[1])(u),$(operators[2])(a)) ⋅ $(operators[3])(v)"
        elseif a_to == 3
            name = "A($(operators[1])(u),$(operators[2])(v)) ⋅ $(operators[3])(a)"
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
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    fixed_argument::Int = 1,
    factor = 1,
    ansatzfunction_operator::Type{<:AbstractFunctionOperator} = Gradient,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0],
    auto_newton::Bool = false,
    quadorder = 0)
````

constructs a trilinearform for a convection term of the form c(a,u,v) = (beta_operator(a)*grad(u),v) where a_from is the id of some unknown of the PDEDescription.
xdim is the space dimension (= number of components of beta_operato(a)) and ncomponents is the number of components of u.
With fixed_argument = 2 a and u can switch their places, i.e.  c(u,a,v) = (beta_operator(u)*grad(a),v),
With auto_newton = true a Newton scheme for a(u,v) = (u*grad(u),v) is automatically derived (and fixed_argument is ignored).

"""
function ConvectionOperator(
    a_from::Int, 
    beta_operator::Type{<:AbstractFunctionOperator},
    xdim::Int,
    ncomponents::Int;
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
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
    function convection_function_fe()
        function closure(result::Array{<:Real,1}, input::Array{<:Real,1})
            for j = 1 : ncomponents
                result[j] = 0
                for k = 1 : xdim
                    result[j] += input[k]*input[xdim+(j-1)*xdim+k]
                end
            end
        end    
    end    
    argsizes = [ncomponents, xdim + ncomponents*xdim]
    if auto_newton
        ## generates a nonlinear form with automatic Newton operators by AD
        if name == "auto"
            name = "(u ⋅ ∇) u ⋅ v"
        end
        return GenerateNonlinearForm(name, [beta_operator, ansatzfunction_operator], [a_from,a_from], testfunction_operator, convection_function_fe,argsizes; ADnewton = true, quadorder = quadorder)     
    else
        ## returns linearised convection operators as a trilinear form (Picard iteration)
        action_kernel = ActionKernel(convection_function_fe(),argsizes; dependencies = "", quadorder = quadorder)
        convection_action = Action(Float64, action_kernel)
        a_to = fixed_argument
        if name == "auto"
            if a_to == 1
                name = "(a ⋅ ∇) u ⋅ v"
            elseif a_to == 2
                name = "(u ⋅ ∇) a ⋅ v"
            elseif a_to == 3
                name = "(u ⋅ ∇) v ⋅ a"
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
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
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
        convection_action = Action(Float64, action_kernel)
        if name == "auto"
            name = "(β × ∇) u ⋅ v"
        end
        O = PDEOperator{Float64, APT_TrilinearForm, AT}(name,[beta_operator,ansatzfunction_operator,testfunction_operator], convection_action, [1,2], factor, regions)
        O.fixed_arguments = [1]
        O.fixed_arguments_ids = [beta]
        O.transposed_assembly = true
        return O
    else
        @error "The rotation form of the convection operator is currently only available in 2D (in 3D please implement it yourself using AbstractTrilinearForm and a user-defined action)"
    end
end





"""
````
function GenerateNonlinearForm(
    name::String,
    operator1::Array{DataType,1},
    coeff_from::Array{Int,1},
    operator2::Type{<:AbstractFunctionOperator},
    action_kernel::Function,
    argsizes::Array{Int,1},
    dim::Int;
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    ADnewton::Bool = false,
    action_kernel_rhs = nothing,
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


Note: this is a highly experimental feature at the moment and will possibly only work when all operators are associated with the same unknown.

can only be applied in PDE LHS
"""
function GenerateNonlinearForm(
    name::String,
    operator1::Array{DataType,1},
    coeff_from::Array{Int,1},
    operator2::Type{<:AbstractFunctionOperator},
    action_kernel::Function,
    argsizes::Array{Int,1};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    ADnewton::Bool = false,
    action_kernel_rhs = nothing,
    quadorder::Int = 0,
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
        result_temp = Vector{Float64}(undef,argsizes[1])
        input_temp = Vector{Float64}(undef,argsizes[3])
        jac_temp = Matrix{Float64}(undef,argsizes[1],argsizes[3])
        Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        cfg =ForwardDiff.JacobianConfig(action_kernel, result_temp, input_temp)
        function newton_kernel(result::Array{<:Real,1}, input_current::Array{<:Real,1}, input_ansatz::Array{<:Real,1})
            ForwardDiff.jacobian!(Dresult, action_kernel, result, input_current, cfg)
            jac = DiffResults.jacobian(Dresult)
            for j = 1 : argsizes[1]
                result[j] = 0
                for k = 1 : argsizes[2]
                    result[j] += jac[j,k] * input_ansatz[k]
                end
            end
            return nothing
        end
        newton_action_kernel = NLActionKernel(newton_kernel, argsizes; dependencies = "", quadorder = quadorder)
        action = Action(Float64, newton_action_kernel)

        # the action for the RHS just evaluates DG and G at input_current
        temp = zeros(Float64, argsizes[1])
        function rhs_kernel(result::Array{<:Real,1}, input_current::Array{<:Real,1})
            fill!(result,0)
            newton_kernel(result, input_current, input_current)
            action_kernel(temp, input_current)
            for j = 1 : argsizes[1]
                result[j] -= temp[j]
            end
            return nothing
        end
        rhs_action_kernel = ActionKernel(rhs_kernel, argsizes; dependencies = "", quadorder = quadorder)
        action_rhs = Action(Float64, rhs_action_kernel)
    else

        # take action_kernel as nonlinear action_kernel
        # = user specifies linearisation of nonlinear operator
        nlform_action_kernel = NLActionKernel(action_kernel, argsizes; dependencies = "", quadorder = quadorder)

        action = Action(Float64, nlform_action_kernel)
        if action_kernel_rhs != nothing
            action_rhs = Action(Float64, action_kernel_rhs)
        else
            action_rhs = nothing
        end
    end

    append!(operator1, [operator2])
    O = PDEOperator{Float64, APT_NonlinearForm, AT}(name, operator1, action, 1:(length(coeff_from)), 1, regions)
    O.fixed_arguments = 1:length(coeff_from)
    O.fixed_arguments_ids = coeff_from
    O.action_rhs = action_rhs === nothing ? NoAction() : action_rhs
    O.transposed_assembly = true
    return O
end


"""
$(TYPEDSIGNATURES)

generates a linearform from an action
    
"""
function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    factor = 1,
    store::Bool = false)

    if name == "auto"
        name = "A($operator(v))"
    end
    PDEOperator{Float64, APT_LinearForm, AT}(name, [operator], action, [1], 1, regions, store, AssemblyAuto)
end


"""
$(TYPEDSIGNATURES)

generates a linearform from a given UserData{<:DataFunction} (whose result dimension has to be 1)
    
"""
function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    regions::Array{Int,1},
    data::UserData{<:AbstractDataFunction};
    name = "auto",
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    factor = 1,
    store::Bool = false)

    if name == "auto"
        name = "$(data.name)⋅$operator(v)"
    end

    ncomponents = data.dimensions[1]
    if typeof(data) <: UserData{AbstractExtendedDataFunction}
        function rhs_function_ext() # result = F(v) = f*operator(v) = f*input
            temp = zeros(Float64,ncomponents)
            function closure(result,input,x, t, region, item, xref)
                eval!(temp, data, x, t, region, item, xref)
                result[1] = 0
                for j = 1 : ncomponents
                    result[1] += temp[j]*input[j] 
                end
                return nothing
            end
        end    
        action_kernel = ActionKernel(rhs_function_ext(),[1, ncomponents]; dependencies = "XTRIL", quadorder = data.quadorder)
    else
        if data.dependencies == "XT"
            function rhs_function_xt() # result = F(v) = f*operator(v) = f*input
                temp = zeros(Float64,ncomponents)
                function closure(result,input,x, t)
                    eval!(temp, data, x, t)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action_kernel = ActionKernel(rhs_function_xt(),[1, ncomponents]; dependencies = "XT", quadorder = data.quadorder)
        elseif data.dependencies == "X"
            function rhs_function_x() # result = F(v) = f*operator(v) = f*input
                temp = zeros(Float64,ncomponents)
                function closure(result,input,x)
                    eval!(temp, data, x, nothing)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action_kernel = ActionKernel(rhs_function_x(),[1, ncomponents]; dependencies = "X", quadorder = data.quadorder)
        elseif data.dependencies == "T"

            function rhs_function_t() # result = F(v) = f*operator(v) = f*input
                temp = zeros(Float64,ncomponents)
                function closure(result,input,t)
                    eval!(temp, data, nothing, t)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action_kernel = ActionKernel(rhs_function_t(),[1, ncomponents]; dependencies = "T", quadorder = data.quadorder)
        else

            function rhs_function_c() # result = F(v) = f*operator(v) = f*input
                temp = zeros(Float64,ncomponents)
                function closure(result,input)
                    eval!(temp, data, nothing, nothing)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                    return nothing
                end
            end    
            action_kernel = ActionKernel(rhs_function_c(),[1, ncomponents]; dependencies = "", quadorder = data.quadorder)
        end
    end
    action = Action(Float64, action_kernel)

    PDEOperator{Float64, APT_LinearForm, AT}(name,[operator], action, [1], 1, regions, store, AssemblyAuto)
end




"""
$(TYPEDSIGNATURES)

constructor for a bilinearform that describes a(u,v) = (beta*grad(u),v) with some user-specified DataFunction beta that writes into
an result array of length ncomponents
    
"""
function ConvectionOperator(
    beta::UserData{AbstractDataFunction},
    ncomponents::Int; 
    name = "auto", 
    store::Bool = false,
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    ansatzfunction_operator::Type{<:AbstractFunctionOperator} = Gradient,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0])

    T = Float64
    xdim = beta.dimensions[1]
    if is_timedependent(beta) && is_xdependent(beta)
        function convection_function_func_xt() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input, x, time)
                # evaluate beta
                eval!(convection_vector,beta,x,time)
                # compute (beta*grad)u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func_xt(), [ncomponents, ncomponents*xdim]; dependencies = "XT", quadorder = beta.quadorder)
    elseif !is_timedependent(beta) && !is_xdependent(beta)
        function convection_function_func() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input)
                # evaluate beta
                eval!(convection_vector,beta, nothing, nothing)
                # compute (beta*grad)u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func(), [ncomponents, ncomponents*xdim]; dependencies = "", quadorder = beta.quadorder)
    elseif !is_timedependent(beta) && is_xdependent(beta)
        function convection_function_func_x() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input, x)
                # evaluate beta
                eval!(convection_vector,beta,x,nothing)
                # compute (beta*grad)u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func_x(), [ncomponents, ncomponents*xdim]; dependencies = "X", quadorder = beta.quadorder)
    elseif is_timedependent(beta) && !is_xdependent(beta)
        function convection_function_func_t() # dot(convection!, input=Gradient)
            convection_vector = zeros(T,xdim)
            function closure(result, input, t)
                # evaluate beta
                eval!(convection_vector,beta,nothing,t)
                # compute (beta*grad)u
                for j = 1 : ncomponents
                    result[j] = 0.0
                    for k = 1 : xdim
                        result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                    end
                end
            end    
        end    
        action_kernel = ActionKernel(convection_function_func_t(), [ncomponents, ncomponents*xdim]; dependencies = "T", quadorder = beta.quadorder)
    end
    if name == "auto"
        name = "(β ⋅ ∇) u ⋅ $testfunction_operator(v)"
    end

    O = PDEOperator{Float64, APT_BilinearForm, AT}(name, [ansatzfunction_operator, testfunction_operator], Action(T, action_kernel), [1], 1, regions, store, AssemblyAuto)
    O.transposed_assembly = true
    return O
end



function update_storage!(O::PDEOperator, CurrentSolution::FEVector, j::Int, k::Int; factor = 1, time::Real = 0)
    @logmsg MoreInfo "Updating storage of PDEOperator $(O.name) in LHS block [$j,$k]"

    set_time!(O.action, time)
    T = typeof(O).parameters[1]
    APT = typeof(O).parameters[2]
    AT = typeof(O).parameters[3]
    if APT <: APT_BilinearForm
        FES = Array{FESpace,1}(undef, 2)
        FES[1] = CurrentSolution[j].FES
        FES[2] = CurrentSolution[k].FES
    else
        @error "No storage functionality available for this operator!"
    end
    O.storage = ExtendableSparseMatrix{Float64,Int64}(FES[1].ndofs,FES[2].ndofs)
    Pattern = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
    assemble!(O.storage, Pattern; transposed_assembly = O.transposed_assembly, factor = factor, skip_preps = false)
    flush!(O.storage)
end

function update_storage!(O::PDEOperator, CurrentSolution::FEVector, j::Int; factor = 1, time::Real = 0)

    @logmsg MoreInfo "Updating storage of PDEOperator $(O.name) in RHS block [$j]"

    set_time!(O.action, time)
    T = typeof(O).parameters[1]
    APT = typeof(O).parameters[2]
    AT = typeof(O).parameters[3]
    if APT <: APT_LinearForm
        FES = Array{FESpace,1}(undef, 1)
        FES[1] = CurrentSolution[j].FES
    else
        @error "No storage functionality available for this operator!"
    end
    O.storage = zeros(Float64,FES[1].ndofs)
    Pattern = AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action_rhs,O.apply_action_to,O.regions)
    assemble!(O.storage, Pattern; factor = factor, skip_preps = false)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock, CurrentSolution) where{T,APT<:APT_BilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, 2)
    FES[1] = O.transposed_assembly ? A.FESY : A.FESX
    FES[2] = O.transposed_assembly ? A.FESX : A.FESY
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock, CurrentSolution::FEVector; non_fixed::Int = 1, fixed_id = 1) where{T,APT<:APT_BilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, 2)
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
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock, CurrentSolution; non_fixed::Int = 1, fixed_id = 1) where{T,APT<:APT_LinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, 1)
    FES[1] = b.FES
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock, CurrentSolution::FEVector) where{T,APT<:APT_TrilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, 3)
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
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end

function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock, CurrentSolution::FEVector; non_fixed::Int = 1, fixed_id = 1) where{T,APT<:APT_TrilinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, 3)
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
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, A::FEMatrixBlock, CurrentSolution::FEVector) where{T,APT<:APT_NonlinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES,A.FESY)
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action,O.apply_action_to,O.regions)
end


function create_assembly_pattern(O::PDEOperator{T,APT,AT}, b::FEVectorBlock, CurrentSolution::FEVector) where{T,APT<:APT_NonlinearForm,AT}
    @debug "Creating assembly pattern for PDEOperator $(O.name)"
    FES = Array{FESpace,1}(undef, length(O.fixed_arguments))
    for a = 1 : length(O.fixed_arguments)
        FES[a] = CurrentSolution[O.fixed_arguments_ids[a]].FES
    end
    push!(FES,b.FES)
    return AssemblyPattern{APT, T, AT}(O.name, FES, O.operators4arguments,O.action_rhs,O.apply_action_to,O.regions)
end


"""
$(TYPEDSIGNATURES)

assembles the operator O into the given FEMatrixBlock A using FESpaces from A. An FEVector CurrentSolution is only needed if the operator involves fixed arguments, e.g. if O is a AbstractTrilinearForm.
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
        assemble!(b, Pattern, CurrentSolution[O.fixed_arguments_ids], skip_preps = skip_preps, factor = O.factor, fixed_arguments = O.fixed_arguments)
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
        if length(O.fixed_arguments_ids) > 0
            assemble!(b, SC.RHS_AssemblyPatterns[j][o],CurrentSolution[O.fixed_arguments_ids]; skip_preps = skip_preps, factor = O.factor*factor)
        else
            assemble!(b, SC.RHS_AssemblyPatterns[j][o]; skip_preps = skip_preps, factor = O.factor*factor)
        end
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




##############################
###### OTHER OPERATORS #######
##############################


struct DiagonalOperator <: AbstractPDEOperator
    name::String
    value::Real
    onlyz::Bool
    regions::Array{Int,1}
end

"""
$(TYPEDSIGNATURES)

puts _value_ on the diagonal entries of the cell dofs within given _regions_

if _onlyz_ == true only values that are zero are changed

can only be applied in PDE LHS
"""
function DiagonalOperator(value::Real = 1.0, onlynz::Bool = true; regions::Array{Int,1} = [0])
    return DiagonalOperator("Diag($value)",value, onlynz, regions)
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



mutable struct FVConvectionDiffusionOperator <: AbstractPDEOperator
    name::String
    diffusion::Float64               # diffusion coefficient
    beta_from::Int                   # component that determines
    fluxes::Array{Float64,2}         # saves normalfluxes of beta here
end


"""
$(TYPEDSIGNATURES)

 finite-volume convection diffusion operator (for cell-wise P0 rho)

 - div(diffusion * grad(rho) + beta rho)

 For diffusion = 0, the upwind divergence: div_upw(beta*rho) is generated
 For diffusion > 0, TODO
                   
"""
function FVConvectionDiffusionOperator(beta_from::Int; diffusion::Float64 = 0.0)
    @assert beta_from > 0
    fluxes = zeros(Float64,0,1)
    return FVConvectionDiffusionOperator("FVConvectionDiffusion",diffusion,beta_from,fluxes)
end


function check_PDEoperator(O::FVConvectionDiffusionOperator, involved_equations)
    return O.beta_from in involved_equations, false, AssemblyAlways
end
function check_PDEoperator(O::CopyOperator, involved_equations)
    return true, true, AssemblyAlways
end


function check_dependency(O::FVConvectionDiffusionOperator, arg::Int)
    return O.beta_from == arg
end




function assemble!(A::FEMatrixBlock, SC, j::Int, k::Int, o::Int,  O::DiagonalOperator, CurrentSolution::FEVector; time::Real = 0)
    @debug "Assembling DiagonalOperator $(O.name)"
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xCellDofs = FE1[CellDofs]
    xCellRegions = FE1.xgrid[CellRegions]
    ncells = num_sources(xCellDofs)
    dof::Int = 0
    for item = 1 : ncells
        for r = 1 : length(O.regions) 
            # check if item region is in regions
            if xCellRegions[item] == O.regions[r] || O.regions[r] == 0
                for k = 1 : num_targets(xCellDofs,item)
                    dof = xCellDofs[k,item]
                    if O.onlyz == true
                        if A[dof,dof] == 0
                            A[dof,dof] = O.value
                        end
                    else
                        A[dof,dof] = O.value
                    end    
                end
            end
        end
    end
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
function assemble!(A::FEMatrixBlock, SC, j::Int, k::Int, o::Int,  O::FVConvectionDiffusionOperator, CurrentSolution::FEVector; time::Real = 0)
    @logmsg MoreInfo "Assembling FVConvectionOperator $(O.name) into matrix"
    T = Float64
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xFaceNodes::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = FE1.xgrid[FaceNodes]
    xFaceNormals::Array{T,2} = FE1.xgrid[FaceNormals]
    xFaceCells::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = FE1.xgrid[FaceCells]
    xFaceVolumes::Array{T,1} = FE1.xgrid[FaceVolumes]
    xCellFaces::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = FE1.xgrid[CellFaces]
    xCellFaceSigns::Union{VariableTargetAdjacency{Int32},Array{Int32,2}} = FE1.xgrid[CellFaceSigns]
    nfaces::Int = num_sources(xFaceNodes)
    ncells::Int = num_sources(xCellFaceSigns)
    nnodes::Int = num_sources(FE1.xgrid[Coordinates])
    
    # ensure that flux field is long enough
    if length(O.fluxes) < nfaces
        O.fluxes = zeros(Float64,1,nfaces)
    end
    # compute normal fluxes of component beta
    c::Int = O.beta_from
    fill!(O.fluxes,0)
    if typeof(SC.LHS_AssemblyPatterns[j,k][o]).parameters[1] <: APT_Undefined
        @debug "Creating assembly pattern for FV convection fluxes $(O.name)"
        SC.LHS_AssemblyPatterns[j,k][o] = ItemIntegrator(Float64, ON_FACES, [NormalFlux]; name = "u ⋅ n")
        evaluate!(O.fluxes,SC.LHS_AssemblyPatterns[j,k][o],[CurrentSolution[c]], skip_preps = false)
    else
        evaluate!(O.fluxes,SC.LHS_AssemblyPatterns[j,k][o],[CurrentSolution[c]], skip_preps = true)
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


function assemble!(b::FEVectorBlock, SC, j::Int, o::Int, O::CopyOperator, CurrentSolution::FEVector; time::Real = 0) 
    for j = 1 : length(b)
        b[j] = CurrentSolution[O.copy_from][j] * O.factor
    end
end



#################################
##### EXTERIOR PDEOPERATORS #####
#################################

### just an idea so far, not usable yet !!!
mutable struct ExteriorPDEOperator{T <: Real} <: AbstractPDEOperator
    name::String
    factor::T
    storage::AbstractMatrix
    init_assemble::Function
    reassembly_trigger::Type{<:AbstractAssemblyTrigger}
    reassemble::Function
end 
