
# type to steer when a PDE block is (re)assembled
abstract type AbstractAssemblyTrigger end
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
# but triggers certain AssemblyPatterns/AbstractActions when called for assembly!
#
# USER-DEFINED ABSTRACTPDEOPERATORS
# might be included if they implement the following interfaces
#
#   (1) to specify what is assembled into the corressponding MatrixBlock:
#       assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractPDEOperatorLHS)
#       assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::AbstractPDEOperatorRHS)
#
#   (2) to allow SolverConfig to check if operator is nonlinear, timedependent:
#       Bool, Bool = check_PDEoperator(O::AbstractPDEOperator)
# 


abstract type AbstractPDEOperator end
abstract type NoConnection <: AbstractPDEOperator end # => empy block in matrix
abstract type AbstractPDEOperatorRHS  <: AbstractPDEOperator end # can be used in RHS (and LHS when one component is fixed)
abstract type AbstractPDEOperatorLHS  <: AbstractPDEOperator end # can be used in RHS (and LHS when one component is fixed)


"""
$(TYPEDEF)

puts _value_ on the diagonal entries of the cell dofs within given _regions_

if _onlyz_ == true only values that are zero are changed

can only be applied in PDE LHS
"""
struct DiagonalOperator <: AbstractPDEOperatorLHS
    name::String
    value::Real
    onlyz::Bool
    regions::Array{Int,1}
end
function DiagonalOperator(value::Real = 1.0, onlynz::Bool = true; regions::Array{Int,1} = [0])
    return DiagonalOperator("Diag($value)",value, onlynz, regions)
end


"""
$(TYPEDEF)

copies entries from TargetVector to rhs block

can only be applied in PDE RHS
"""
struct CopyOperator <: AbstractPDEOperatorRHS
    name::String
    copy_from::Int
    factor::Real
end
function CopyOperator(copy_from, factor)
    return CopyOperator("CopyOperator",copy_from, factor)
end

mutable struct AbstractBilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    apply_action_to::Int
    regions::Array{Int,1}
    transposed_assembly::Bool
    store_operator::Bool                    # should the matrix repsentation of the operator be stored?
    storage::AbstractArray{<:Real,2}  # matrix can be stored here to allow for fast matmul operations in iterative settings
end

"""
````
function AbstractBilinearForm(name,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    apply_action_to = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)
````

abstract bilinearform operator that assembles
- b(u,v) = int_regions action(operator1(u)) * operator2(v) if apply_action_to = 1
- b(u,v) = int_regions operator1(u) * action(operator2(v)) if apply_action_to = 2

can only be applied in PDE LHS
"""
function AbstractBilinearForm(name,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    action::AbstractAction;
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    apply_action_to = 1,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)
    return AbstractBilinearForm{AT}(name,operator1, operator2, action, apply_action_to, regions,transposed_assembly,false,zeros(Float64,0,0))
end
function AbstractBilinearForm(operator1,operator2; apply_action_to = 1, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("$operator1 x $operator2",operator1, operator2, DoNotChangeAction(1); apply_action_to = apply_action_to, regions = regions)
end

"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (kappa * nabla u, nabla v) where kappa is some constant diffusion coefficient
"""
function LaplaceOperator(diffusion::Real = 1.0, xdim::Int = 2, ncomponents::Int = 1; gradient_operator = Gradient, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("Laplacian",gradient_operator, gradient_operator, MultiplyScalarAction(diffusion, ncomponents*xdim); regions = regions)
end

"""
$(TYPEDSIGNATURES)

constructor for AbstractBilinearForm that describes a(u,v) = (C grad(u), grad(v)) where C is the 1D stiffness tensor
C grad(u) = mu grad(u)
    
"""
function HookStiffnessOperator1D(mu::Real; regions::Array{Int,1} = [0], gradient_operator = TangentialGradient)
    function tensor_apply_1d(result, input)
        # just Hook law like a spring where mu is the elasticity modulus
        result[1] = mu*input[1]
    end   
    action_kernel = ActionKernel(tensor_apply_1d, [1,1]; dependencies = "", quadorder = 0)
    action = Action(Float64, action_kernel)
    return AbstractBilinearForm("Hookian1D",gradient_operator, gradient_operator, action; regions = regions)
end

"""
$(TYPEDSIGNATURES)

constructor for AbstractBilinearForm that describes a(u,v) = (C eps(u), eps(v)) where C is the 3D stiffness tensor
for isotropic media in Voigt notation, i.e.
C eps(u) = 2 mu eps(u) + lambda tr(eps(u)) for Lame parameters mu and lambda
    
    In Voigt notation C is a 3 x 3 matrix
    C = [c11,c12,  0
         c12,c11,  0
           0,  0,c33]
    
    where c33 = shear_modulus, c12 = lambda and c11 = 2*c33 + c12
    
"""
function HookStiffnessOperator2D(mu::Real, lambda::Real; regions::Array{Int,1} = [0], gradient_operator = SymmetricGradient)
    function tensor_apply_2d(result, input)
        result[1] = (lambda + 2*mu)*input[1] + lambda*input[2]
        result[2] = (lambda + 2*mu)*input[2] + lambda*input[1]
        result[3] = mu*input[3]
    end   
    action_kernel = ActionKernel(tensor_apply_2d, [3,3]; dependencies = "", quadorder = 0)
    action = Action(Float64, action_kernel)
    return AbstractBilinearForm("Hookian2D",gradient_operator, gradient_operator, action; regions = regions)
end

"""
$(TYPEDSIGNATURES)

constructor for AbstractBilinearForm that describes a(u,v) = (C eps(u), eps(v)) where C is the 3D stiffness tensor
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
function HookStiffnessOperator3D(mu::Real, lambda::Real; regions::Array{Int,1} = [0], gradient_operator = SymmetricGradient)
    function tensor_apply_3d(result, input)
        result[1] = (lambda + 2*mu)*input[1] + lambda*(input[2] + input[3])
        result[2] = (lambda + 2*mu)*input[2] + lambda*(input[1] + input[3])
        result[3] = (lambda + 2*mu)*input[3] + lambda*(input[1] + input[2])
        result[4] = mu*input[4]
        result[5] = mu*input[5]
        result[6] = mu*input[6]
    end   
    action_kernel = ActionKernel(tensor_apply_3d, [6,6]; dependencies = "", quadorder = 0)
    action = Action(Float64, action_kernel)
    return AbstractBilinearForm("Hookian3D", gradient_operator, gradient_operator, action; regions = regions)
end


"""
$(TYPEDSIGNATURES)

constructor for AbstractBilinearForm that describes a(u,v) = (A(u),v) or (u,A(v)) with some user-specified action A
    
"""
function ReactionOperator(action::AbstractAction; apply_action_to = 1, identity_operator = Identity, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("Reaction",identity_operator, identity_operator, action; apply_action_to = apply_action_to, regions = regions)
end

"""
$(TYPEDSIGNATURES)

constructor for AbstractBilinearForm that describes a(u,v) = (beta*grad(u),v) with some user-specified function beta with the
interface beta(result,x::Array{<:Real,1}) (so it writes its result into result and returns nothing)
    
"""
function ConvectionOperator(T::Type{<:Real}, beta::UserData{AbstractDataFunction}, ncomponents::Int; testfunction_operator::Type{<:AbstractFunctionOperator} = Identity, regions::Array{Int,1} = [0])
    xdim = beta.dimensions[1]
    function convection_function_func() # dot(convection!, input=Gradient)
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
    action_kernel = ActionKernel(convection_function_func(), [ncomponents, ncomponents*xdim]; name = "L2 error kernel", dependencies = "XT", quadorder = beta.quadorder)
    return AbstractBilinearForm("($(beta.name) * Gradient) u * v", Gradient,testfunction_operator, Action(T, action_kernel); regions = regions, transposed_assembly = true)
end


"""
````
mutable struct AbstractMultilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operators::Array{DataType,1}
    action::AbstractAction
    regions::Array{Int,1}
end
````

abstract multi-linearform with arbitrary many argument of the form

m(v1,v2,...,vk) = (A(O(v1),O(v2),...,O(vk-1)),Ok(vk))

(so far only intended for use as RHSOperator together with MLF2RHS)
"""
mutable struct AbstractMultilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operators::Array{DataType,1}
    action::AbstractAction
    regions::Array{Int,1}
end


mutable struct AbstractNonlinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operator1::Array{DataType,1}
    coeff_from::Array{Int,1}     # unknown id where coefficient for each operator in operator1 are taken from
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    action_rhs
    regions::Array{Int,1}
    ADnewton::Bool
    transposed_assembly::Bool
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
    action_kernel::UserData{<:AbstractActionKernel};
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    ADnewton::Bool = false,
    action_kernel_rhs = nothing,
    bonus_quadorder::Int = 0,
    regions = [0])

    argsizes::Array{Int,1} = action_kernel.dimensions
    ### Newton scheme for a nonlinear operator G(u) is
    ## seek u_next such that : DG u_next = DG u - G(u)
    if ADnewton
        name = name * " [AD-Newton]"
        # the action for the derivative matrix DG is calculated by automatic differentiation (AD)
        # from the given action_kernel of the operator G

        # for differentation other dependencies of the action_kernel are fixed
        result_temp = Vector{Float64}(undef,argsizes[1])
        input_temp = Vector{Float64}(undef,argsizes[2])
        jac_temp = Matrix{Float64}(undef,argsizes[1],argsizes[2])
        Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        cfg =ForwardDiff.JacobianConfig(action_kernel.user_function, result_temp, input_temp)
        function newton_kernel(result, input_current, input_ansatz)
            ForwardDiff.jacobian!(Dresult, action_kernel.user_function, result, input_current, cfg)
            jac = DiffResults.jacobian(Dresult)
            
            for j = 1 : argsizes[1]
                result[j] = 0
                for k = 1 : argsizes[2]
                    result[j] += jac[j,k] * input_ansatz[k]
                end
            end
            return nothing
        end
        newton_action_kernel = NLActionKernel(newton_kernel, argsizes; dependencies = "", quadorder = action_kernel.quadorder)
        action = Action(Float64, newton_action_kernel)

        # the action for the RHS just evaluates DG and G at input_current
        temp = zeros(Float64, argsizes[1])
        function rhs_kernel(result, input_current)
            fill!(result,0)
            newton_kernel(result, input_current, input_current)
            action_kernel.user_function(temp, input_current)
            for j = 1 : argsizes[1]
                result[j] -= temp[j]
            end
            return nothing
        end
        rhs_action_kernel = ActionKernel(rhs_kernel, argsizes; dependencies = "", quadorder = action_kernel.quadorder)
        action_rhs = Action(Float64, rhs_action_kernel)
    else
        action = Action(Float64, action_kernel)
        if action_kernel_rhs != nothing
            action_rhs = Action(Float64, action_kernel_rhs)
        else
            action_rhs = nothing
        end
    end

    return AbstractNonlinearForm{AT}(name, operator1, coeff_from, operator2, action, action_rhs, regions, ADnewton, true)
end




"""
$(TYPEDEF)

considers the second argument to be a Lagrange multiplier for operator(first argument) = 0,
automatically triggers copy of transposed operator in transposed block, hence only needs to be assigned and assembled once!

can only be applied in PDE LHS
"""
struct LagrangeMultiplier <: AbstractPDEOperatorLHS
    name::String
    operator::Type{<:AbstractFunctionOperator} # e.g. Divergence, automatically aligns with transposed block
end
function LagrangeMultiplier(operator::Type{<:AbstractFunctionOperator})
    return LagrangeMultiplier("LagrangeMultiplier($operator)",operator)
end



"""
````
mutable struct AbstractTrilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operator1::Type{<:AbstractFunctionOperator} # operator for argument 1
    operator2::Type{<:AbstractFunctionOperator} # operator for argument 1
    operator3::Type{<:AbstractFunctionOperator} # operator for argument 1
    a_from::Int     # unknown id where fixed argument takes its values from
    a_to::Int       # position of fixed argument
    action::AbstractAction # is applied to argument 1 and 2
    regions::Array{Int,1}
    transposed_assembly::Bool
end
````

abstract trilinearform operator that assembles
- c(a,u,v) = int_regions action(operator1(a) * operator2(u))*operator3(v)   (if a_to = 1)
- c(u,a,v) = int_regions action(operator1(u) * operator2(a))*operator3(v)   (if a_to = 2)

where a_from is the id of one of the unknowns of the PDEsystem

can only be applied in PDE LHS
"""
mutable struct AbstractTrilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operator1::Type{<:AbstractFunctionOperator} # operator for argument 1
    operator2::Type{<:AbstractFunctionOperator} # operator for argument 2
    operator3::Type{<:AbstractFunctionOperator} # operator for argument 3
    a_from::Int     # unknown id where fixed argument takes its values from
    a_to::Int       # position of fixed argument
    action::AbstractAction # is applied to argument 1 and 2
    regions::Array{Int,1}
    transposed_assembly::Bool
end
function AbstractTrilinearForm(name,
    operator1::Type{<:AbstractFunctionOperator},
    operator2::Type{<:AbstractFunctionOperator},
    operator3::Type{<:AbstractFunctionOperator},
    a_from::Int,
    a_to::Int,
    action::AbstractAction;
    AT::Type{<:AbstractAssemblyType} = ON_CELLS,
    regions::Array{Int,1} = [0],
    transposed_assembly::Bool = false)
    return AbstractTrilinearForm{AT}(name, operator1, operator2, operator3, a_from, a_to, action, regions,transposed_assembly)
end

"""
````
function ConvectionOperator(
    a_from::Int, 
    beta_operator,
    xdim::Int,
    ncomponents::Int;
    fixed_argument::Int = 1,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0],
    auto_newton::Bool = false)
````

constructs an PDE operator for a convection term of the form c(a,u,v) = (beta_operator(a)*grad(u),v) where a_from is the id of some unknown of the PDEDescription.
xdim is the space dimension (= number of components of beta_operato(a)) and ncomponents is the number of components of u.
With fixed_argument = 2 a and u can switch their places, i.e.  c(u,a,v) = (beta_operator(u)*grad(a),v). 
With auto_newton = true a Newton scheme for a(u,v) = (u*grad(u),v) is automatically derived (and fixed_argument is ignored).

"""
function ConvectionOperator(
    a_from::Int, 
    beta_operator,
    xdim::Int,
    ncomponents::Int;
    fixed_argument::Int = 1,
    testfunction_operator::Type{<:AbstractFunctionOperator} = Identity,
    regions::Array{Int,1} = [0],
    auto_newton::Bool = false)

    # action input consists of two inputs
    # input[1:xdim] = operator1(a)
    # input[xdim+1:end] = grad(u)
    function convection_function_fe()
        function closure(result, input)
            for j = 1 : ncomponents
                result[j] = 0.0
                for k = 1 : xdim
                    result[j] += input[k]*input[xdim+(j-1)*xdim+k]
                end
            end
        end    
    end    
    action_kernel = ActionKernel(convection_function_fe(),[ncomponents, xdim + ncomponents*xdim]; dependencies = "", quadorder = 0)
    if auto_newton
        ## generates a nonlinear form with automatic Newton operators by AD
        return GenerateNonlinearForm("(u * grad) u  * v", [beta_operator, Gradient], [a_from,a_from], testfunction_operator, action_kernel; ADnewton = true)     
    else
        ## returns linearised convection operators as a trilinear form (Picard iteration)
        convection_action = Action(Float64, action_kernel)
        a_to = fixed_argument
        if a_to == 1
            name = "(a(=unknown $(a_from)) * Gradient) u * v"
        elseif a_to == 2
            name = "(u * Gradient) a(=unknown $(a_from)) * v"
        elseif a_to == 3
            name = "(u * Gradient) v * a(=unknown $(a_from))"
        end
        
        return AbstractTrilinearForm{ON_CELLS}(name,beta_operator,Gradient,testfunction_operator,a_from,a_to,convection_action, regions, true)
    end

end

"""
$(TYPEDSIGNATURES)

constructor for AbstractBilinearForm that describes a(u,v) = (beta x curl(u),v)
where beta is the id of some unknown vector field of the PDEDescription, u and v
are also vector-fields and x is the cross product (so far this is only implemented in 2D)
    
"""
function ConvectionRotationFormOperator(beta::Int, beta_operator::Type{<:AbstractFunctionOperator}, xdim::Int, ncomponents::Int; testfunction_operator::Type{<:AbstractFunctionOperator} = Identity, regions::Array{Int,1} = [0])
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
    return AbstractTrilinearForm{ON_CELLS}("(a(=unknown $beta) x Curl2D u ) * v",beta_operator,Curl2D,testfunction_operator,beta, 1, convection_action, regions, true)
end


"""
````
mutable struct RhsOperator{AT<:AbstractAssemblyType} <: AbstractPDEOperatorRHS
    name::String
    data::UserData{AbstractDataFunction}
    testfunction_operator::Type{<:AbstractFunctionOperator}
    regions::Array{Int,1}
    store_operator::Bool               # should the matrix representation of the operator be stored?
    storage::AbstractArray{Float64,1}  # matrix can be stored here to allow for fast matmul operations in iterative settings
end
````

right-hand side operator

can only be applied in PDE RHS
"""
mutable struct RhsOperator{AT<:AbstractAssemblyType} <: AbstractPDEOperatorRHS
    name::String
    data::UserData{<:AbstractDataFunction}
    testfunction_operator::Type{<:AbstractFunctionOperator}
    regions::Array{Int,1}
    store_operator::Bool               # should the matrix representation of the operator be stored?
    storage::AbstractArray{Float64,1}  # matrix can be stored here to allow for fast matmul operations in iterative settings
end

function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    regions::Array{Int,1},
    data::UserData{<:AbstractDataFunction};
    name = "auto",
    on_boundary::Bool = false)

    if name == "auto"
        name = "$(data.name) * $operator(v_h)"
    end
    if on_boundary == true
        return RhsOperator{ON_BFACES}(name, data, operator, regions, false, [])
    else
        return RhsOperator{ON_CELLS}(name, data, operator, regions, false, [])
    end
end


"""
````
struct BLF2RHS <: AbstractPDEOperatorRHS
    name::String
    BLF::AbstractBilinearForm
    data_id::Int
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end
````

evaluation of a AbstractBilinearForm BLF (multiplied by a factor) where the second argument is fixed by the given FEVectorBlock of the current solution coressponding to the given data_id.

The operator must be manually marked as nonlinear or time-dependent to trigger reassembly at each iteration or each timestep.

can only be applied in PDE RHS
"""
struct BLF2RHS <: AbstractPDEOperatorRHS
    name::String
    BLF::AbstractBilinearForm
    data_id::Int
    factor::Real
    fixed_argument::Int
    nonlinear::Bool
    timedependent::Bool
end

function BLF2RHS(BLF, data_id, factor; name = "auto", fixed_argument::Int = 2, nonlinear::Bool = false, timedependent::Bool = false)
    if name == "auto"
        if fixed_argument == 1
            name = "BLF($(BLF.name)(#$(data_id),*)"
        elseif fixed_argument == 2
            name = "BLF($(BLF.name)(*,#$(data_id))"
        end
    end
    return BLF2RHS(name, BLF, data_id, factor, fixed_argument, nonlinear, timedependent)
end

"""
````
struct TLF2RHS <: AbstractPDEOperatorRHS
    name::String
    TLF::AbstractTrilinearForm
    data_ids::Array{Int,1}
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end
````

evaluation of a AbstractTrilinearForm TLF (multiplied by a factor) where the first and second argument are fixed by the FEVectorBlocks of the current solution coressponding to the given data_ids.

The operator must be manually marked as nonlinear or time-dependent to trigger reassembly at each iteration or each timestep.

can only be applied in PDE RHS
"""
struct TLF2RHS <: AbstractPDEOperatorRHS
    name::String
    TLF::AbstractTrilinearForm
    data_ids::Array{Int,1}
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end

function TLF2RHS(TLF, data_ids, factor::Real = 1; name = "auto", nonlinear::Bool = false, timedependent::Bool = false)
    if name == "auto"
        name = "TLF($(TLF.name))(#$(data_ids[1]),#$(data_ids[2]), *)"
    end
    return TLF2RHS(name, TLF, data_ids, factor, nonlinear, timedependent)
end



"""
````
struct MLF2RHS <: AbstractPDEOperatorRHS
    name::String
    MLF::AbstractMultilinearForm
    data_ids::Array{Int,1}
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end
````

evaluation of a AbstractMultilinearForm MLF (multiplied by a factor) where all but the last argument are fixed by the FEVectorBlocks of the current solution coressponding to the given data_ids.

The operator must be manually marked as nonlinear or time-dependent to trigger reassembly at each iteration or each timestep.

can only be applied in PDE RHS
"""
struct MLF2RHS <: AbstractPDEOperatorRHS
    name::String
    MLF::AbstractMultilinearForm
    data_ids::Array{Int,1}
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end

function MLF2RHS(MLF, data_ids, factor; name = "auto", nonlinear::Bool = false, timedependent::Bool = false)
    if name == "auto"
        name = "MLF2RHS($(MLF.name))"
    end
    return MLF2RHS(name, MLF, data_ids, factor, nonlinear, timedependent)
end


#####################################
### FVConvectionDiffusionOperator ###
#####################################
#
# finite-volume convection diffusion operator (for cell-wise P0 rho)
#
# - div(\kappa \nabla \rho + beta*rho)
#
# For kappa = 0, the upwind divergence: div_upw(beta*rho) is generated
# For kappa > 0, TODO
# 
# (1) calculate normalfluxes from component at _beta_from_
# (2) compute FV flux on each face and put coefficients on neighbouring cells in matrix
#
#     if kappa == 0:
#           div_upw(beta*rho)|_T = sum_{F face of T} normalflux(F) * rho(F)
#
#           where rho(F) is the rho in upwind direction 
#
#     and put it into P0xP0 matrix block like this:
#
#           Loop over cell, face of cell
#
#               other_cell = other face neighbour cell
#               if flux := normalflux(F_j) * CellFaceSigns[face,cell] > 0
#                   A(cell,cell) += flux
#                   A(other_cell,cell) -= flux
#               else
#                   A(other_cell,other_cell) -= flux
#                   A(cell,other_cell) += flux
#                   
# see coressponding assemble! routine

mutable struct FVConvectionDiffusionOperator <: AbstractPDEOperatorLHS
    name::String
    diffusion::Float64               # diffusion coefficient
    beta_from::Int                   # component that determines
    fluxes::Array{Float64,2}         # saves normalfluxes of beta here
end
function FVConvectionDiffusionOperator(beta_from::Int; diffusion::Float64 = 0.0)
    @assert beta_from > 0
    fluxes = zeros(Float64,0,1)
    return FVConvectionDiffusionOperator("FVConvectionDiffusion",diffusion,beta_from,fluxes)
end
function check_PDEoperator(O::RhsOperator)
    return false, is_timedependent(O.data)
end


################ ASSEMBLY SPECIFICATIONS ################



# check if operator causes nonlinearity or time-dependence
function check_PDEoperator(O::AbstractPDEOperator)
    return false, false
end
function check_PDEoperator(O::AbstractBilinearForm)
    try
        return false, is_timedependent(O.action.kernel)
    catch
        return false, false
    end
end
function check_PDEoperator(O::AbstractTrilinearForm)
    try
        return true, is_timedependent(O.action.kernel)
    catch
        return true, false
    end
end
function check_PDEoperator(O::FVConvectionDiffusionOperator)
    return O.beta_from != 0, false
end
function check_PDEoperator(O::CopyOperator)
    return true, true
end
function check_PDEoperator(O::BLF2RHS)
    return O.nonlinear, O.timedependent
end
function check_PDEoperator(O::TLF2RHS)
    return O.nonlinear, O.timedependent
end
function check_PDEoperator(O::MLF2RHS)
    return O.nonlinear, O.timedependent
end
function check_PDEoperator(O::AbstractNonlinearForm)
    try
        return true, is_timedependent(O.action.kernel)
    catch
        return true, false
    end
end

# check if operator also depends on arg (additional to the argument relative to position in PDEDescription)
function check_dependency(O::AbstractPDEOperator, arg::Int)
    return false
end

function check_dependency(O::FVConvectionDiffusionOperator, arg::Int)
    return O.beta_from == arg
end

function check_dependency(O::AbstractTrilinearForm, arg::Int)
    return O.a_from == arg
end
function check_dependency(O::AbstractNonlinearForm, arg::Int)
    return arg in O.coeff_from
end

# check if operator on the LHS also needs to modify the RHS
function LHSoperator_also_modifies_RHS(O::AbstractPDEOperator)
    return false
end
function LHSoperator_also_modifies_RHS(O::AbstractNonlinearForm)
    return O.action_rhs != nothing
end




function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::DiagonalOperator; time::Real = 0, verbosity::Int = 0)
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


function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::FVConvectionDiffusionOperator; time::Real = 0, verbosity::Int = 0)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xFaceNodes = FE1.xgrid[FaceNodes]
    xFaceNormals = FE1.xgrid[FaceNormals]
    xFaceCells = FE1.xgrid[FaceCells]
    xFaceVolumes = FE1.xgrid[FaceVolumes]
    xCellFaces = FE1.xgrid[CellFaces]
    xCellFaceSigns = FE1.xgrid[CellFaceSigns]
    nfaces = num_sources(xFaceNodes)
    ncells = num_sources(xCellFaceSigns)
    nnodes = num_sources(FE1.xgrid[Coordinates])
    
    # ensure that flux field is long enough
    if length(O.fluxes) < nfaces
        O.fluxes = zeros(Float64,1,nfaces)
    end
    # compute normal fluxes of component beta
    c = O.beta_from
    fill!(O.fluxes,0)
    fluxIntegrator = ItemIntegrator{Float64,ON_FACES}(NormalFlux, DoNotChangeAction(1), [0])
    evaluate!(O.fluxes,fluxIntegrator,[CurrentSolution[c]]; verbosity = verbosity - 1)

    nfaces4cell = 0
    face = 0
    flux = 0.0
    other_cell = 0
    for cell = 1 : ncells
        nfaces4cell = num_targets(xCellFaces,cell)
        for cf = 1 : nfaces4cell
            face = xCellFaces[cf,cell]
            other_cell = xFaceCells[1,face]
            if other_cell == cell
                other_cell = xFaceCells[2,face]
            end
            flux = O.fluxes[face] * xCellFaceSigns[cf,cell] # sign okay?
            if (other_cell > 0) 
                flux *= 1 // 2 # because it will be accumulated on two cells
            end       
            if flux > 0 # flow from cell to other_cell
                A[cell,cell] += flux
                if other_cell > 0
                    A[other_cell,cell] -= flux
                    # otherwise flow goes out of domain
                end    
            else # flow from other_cell into cell
                if other_cell > 0 # flow comes from neighbour cell
                    A[other_cell,other_cell] -= flux
                    A[cell,other_cell] += flux
                else # flow comes from outside domain
                   #  A[cell,cell] += flux
                end 
            end
        end
    end
end



function update_storage!(O::AbstractBilinearForm{AT}, CurrentSolution::FEVector, j::Int, k::Int; factor = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}

    if verbosity > 0
        println("  Updating storage of operator $(O.name)")
    end

    # ensure that storage is large_enough
    FE1 = CurrentSolution[j].FES
    FE2 = CurrentSolution[k].FES
    O.storage = ExtendableSparseMatrix{Float64,Int32}(FE1.ndofs,FE2.ndofs)

    if FE1 == FE2 && O.operator1 == O.operator2
        BLF = SymmetricBilinearForm(Float64, AT, FE1, O.operator1, O.action; regions = O.regions)    
    else
        BLF = BilinearForm(Float64, AT, FE1, FE2, O.operator1, O.operator2, O.action; regions = O.regions)    
    end

    assemble!(O.storage, BLF; apply_action_to = O.apply_action_to, factor = factor, verbosity = verbosity)
    flush!(O.storage)
end


function update_storage!(O::RhsOperator{AT}, CurrentSolution::FEVector, j::Int; factor = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}

    if verbosity > 0
        println("  Updating storage of RhsOperator")
    end

    # ensure that storage is large_enough
    FE = CurrentSolution[j].FES
    O.storage = zeros(Float64,FE.ndofs)
    ncomponents = O.data.dimensions[1]

    function rhs_function() # result = F(v) = f*operator(v) = f*input
        temp = zeros(Float64,ncomponents)
        function closure(result,input,x)
            eval!(temp, O.data, x, time)
            result[1] = 0
            for j = 1 : ncomponents
                result[1] += temp[j]*input[j] 
            end
        end
    end    
    action_kernel = ActionKernel(rhs_function(),[1, ncomponents]; dependencies = "X", quadorder = O.data.quadorder)
    action = Action(Float64, action_kernel)
    RHS = LinearForm(Float64,AT, FE, O.testfunction_operator, action; regions = O.regions)
    assemble!(O.storage, RHS; factor = factor, verbosity = verbosity)
end


function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractBilinearForm{AT}; factor = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    if O.store_operator == true 
        addblock!(A,O.storage; factor = factor)
    else
        FE1 = A.FESX
        FE2 = A.FESY
        if FE1 == FE2 && O.operator1 == O.operator2
            BLF = SymmetricBilinearForm(Float64, AT, FE1, O.operator1, O.action; regions = O.regions)    
        else
            BLF = BilinearForm(Float64, AT, FE1, FE2, O.operator1, O.operator2, O.action; regions = O.regions)    
        end
        set_time!(O.action, time)
        assemble!(A, BLF; apply_action_to = O.apply_action_to, factor = factor, verbosity = verbosity, transposed_assembly = O.transposed_assembly)
    end
end


function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::AbstractBilinearForm{AT}; factor = 1, time::Real = 0, verbosity::Int = 0, fixed_component::Int = 0) where {AT<:AbstractAssemblyType}
    if O.store_operator == true
        addblock_matmul!(b,O.storage,CurrentSolution[fixed_component]; factor = factor)
    else
        FE1 = b.FES
        FE2 = CurrentSolution[fixed_component].FES
        if FE1 == FE2 && O.operator1 == O.operator2
            BLF = SymmetricBilinearForm(Float64, AT, FE1, O.operator1, O.action; regions = O.regions)    
        else
            BLF = BilinearForm(Float64, AT, FE1, FE2, O.operator1, O.operator2, O.action; regions = O.regions)    
        end
        set_time!(O.action, time)
        assemble!(b, CurrentSolution[fixed_component], BLF; apply_action_to = O.apply_action_to, factor = factor, verbosity = verbosity)
    end
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::TLF2RHS; factor = 1, time::Real = 0, verbosity::Int = 0)
    FE1 = CurrentSolution[O.data_ids[1]].FES
    FE2 = CurrentSolution[O.data_ids[1]].FES
    FE3 = b.FES
    TLF = TrilinearForm(Float64, typeof(O.TLF).parameters[1], FE1, FE2, FE3, O.TLF.operator1, O.TLF.operator2, O.TLF.operator3, O.TLF.action; regions = O.TLF.regions)  
    set_time!(O.TLF.action, time)
    assemble!(b, CurrentSolution[O.data_ids[1]], CurrentSolution[O.data_ids[2]], TLF; factor = factor * O.factor, verbosity = verbosity)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::MLF2RHS; factor = 1, time::Real = 0, verbosity::Int = 0)
    FES = []
    for k = 1 : length(O.Data)
        push!(FES, CurrentSolution[O.data_ids[k]].FES)
    end
    push!(FES, b.FES)
    FES = Array{FESpace,1}(FES)
    MLF = MultilinearForm(Float64, typeof(O.MLF).parameters[1], FES, O.MLF.operators, O.MLF.action; regions = O.MLF.regions)  
    set_time!(O.MLF.action, time)
    assemble!(b, CurrentSolution[O.data_ids], MLF; factor = factor * O.factor, verbosity = verbosity)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::BLF2RHS; factor = 1, time::Real = 0, verbosity::Int = 0)
    if O.BLF.store_operator == true
        addblock_matmul!(b,O.BLF.storage,CurrentSolution[O.data_id]; factor = factor)
    else
        FE1 = b.FES
        FE2 = CurrentSolution(O.data_id).FES
        if FE1 == FE2 && O.BLF.operator1 == O.BLF.operator2
            BLF = SymmetricBilinearForm(Float64, typeof(O.BLF).parameters[1], FE1, O.BLF.operator1, O.BLF.action; regions = O.BLF.regions)    
        else
            BLF = BilinearForm(Float64, typeof(O.BLF).parameters[1], FE1, FE2, O.BLF.operator1, O.BLF.operator2, O.BLF.action; regions = O.BLF.regions)    
        end
        set_time!(O.BLF.action, time)
        assemble!(b, CurrentSolution[O.data_id], BLF; apply_action_to = O.BLF.apply_action_to, factor = factor * O.factor, verbosity = verbosity, fixed_argument = O.fixed_argument)
    end
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractTrilinearForm; time::Real = 0, verbosity::Int = 0)
    FE1 = CurrentSolution[O.a_from].FES
    FE2 = A.FESX
    FE3 = A.FESY
    TLF = TrilinearForm(Float64, typeof(O).parameters[1], FE1, FE2, FE3, O.operator1, O.operator2, O.operator3, O.action; regions = O.regions)  
    set_time!(TLF.action, time)
    assemble!(A, CurrentSolution[O.a_from], TLF; verbosity = verbosity, fixed_argument = O.a_to, transposed_assembly = O.transposed_assembly)
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::LagrangeMultiplier; time::Real = 0, verbosity::Int = 0, At::FEMatrixBlock)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert At.FESX == FE2
    @assert At.FESY == FE1
    DivPressure = BilinearForm(Float64, ON_CELLS, FE1, FE2, O.operator, Identity, MultiplyScalarAction(-1.0,1))   
    assemble!(A, DivPressure; verbosity = verbosity, transpose_copy = At)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::RhsOperator{AT}; factor = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    if O.store_operator == true
        addblock!(b, O.storage; factor = factor)
    else
        FE = b.FES
        ncomponents = O.data.dimensions[1]
        if typeof(O.data) <: UserData{AbstractExtendedDataFunction}
            function rhs_function_ext() # result = F(v) = f*operator(v) = f*input
                temp = zeros(Float64,ncomponents)
                function closure(result,input,x, region, item, xref)
                    eval!(temp, O.data, x, time, region, item, xref)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                end
            end    
            action_kernel = ActionKernel(rhs_function_ext(),[1, ncomponents]; dependencies = "XRIL", quadorder = O.data.quadorder)
        else
            function rhs_function() # result = F(v) = f*operator(v) = f*input
                temp = zeros(Float64,ncomponents)
                function closure(result,input,x)
                    eval!(temp, O.data, x, time)
                    result[1] = 0
                    for j = 1 : ncomponents
                        result[1] += temp[j]*input[j] 
                    end
                end
            end    
            action_kernel = ActionKernel(rhs_function(),[1, ncomponents]; dependencies = "X", quadorder = O.data.quadorder)
        end
        action = Action(Float64, action_kernel)
        RHS = LinearForm(Float64,AT, FE, O.testfunction_operator, action; regions = O.regions)
        assemble!(b, RHS; factor = factor, verbosity = verbosity)
    end
end


function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::CopyOperator; time::Real = 0, verbosity::Int = 0) 
    for j = 1 : length(b)
        b[j] = CurrentSolution[O.copy_from][j] * O.factor
    end
end




function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractNonlinearForm; time::Real = 0, verbosity::Int = 0)
    FE = Array{FESpace,1}(undef, length(O.coeff_from))
    for j = 1 : length(O.coeff_from)
        FE[j] = CurrentSolution[O.coeff_from[j]].FES
    end
    FE2 = A.FESY
    NLF = NonlinearForm(Float64, ON_CELLS, FE, FE2, O.operator1, O.operator2, O.action; regions = O.regions)  
    set_time!(O.action, time)
    assemble!(A, NLF, CurrentSolution[O.coeff_from]; verbosity = verbosity, transposed_assembly = O.transposed_assembly)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::AbstractNonlinearForm; time::Real = 0, verbosity::Int = 0)
    FE = Array{FESpace,1}(undef, length(O.coeff_from))
    for j = 1 : length(O.coeff_from)
        FE[j] = CurrentSolution[O.coeff_from[j]].FES
    end
    FE2 = b.FES
    NLF = NonlinearForm(Float64, ON_CELLS, FE, FE2, O.operator1, O.operator2, O.action_rhs; regions = O.regions)  
    set_time!(O.action_rhs, time)
    assemble!(b, NLF, CurrentSolution[O.coeff_from]; verbosity = verbosity)
end
