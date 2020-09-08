
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

"""
$(TYPEDEF)

abstract bilinearform operator that assembles
- b(u,v) = int_regions action(operator1(u)) * operator2(v) if apply_action_to = 1
- b(u,v) = int_regions operator1(u) * action(operator2(v)) if apply_action_to = 2

can only be applied in PDE LHS
"""
mutable struct AbstractBilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    apply_action_to::Int
    regions::Array{Int,1}
    transposed_assembly::Bool
    store_operator::Bool                    # should the matrix repsentation of the operator be stored?
    storage::AbstractArray{Float64,2}  # matrix can be stored here to allow for fast matmul operations in iterative settings
end
function AbstractBilinearForm(name, operator1,operator2, action; apply_action_to = 1, regions::Array{Int,1} = [0], transposed_assembly::Bool = false)
    return AbstractBilinearForm{ON_CELLS}(name,operator1, operator2, action, apply_action_to, regions,transposed_assembly,false,zeros(Float64,0,0))
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
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (C grad(u), grad(v)) where C is the 1D stiffness tensor
C grad(u) = mu grad(u)
    
"""
function HookStiffnessOperator1D(mu::Real; regions::Array{Int,1} = [0], gradient_operator = TangentialGradient)
    function tensor_apply_1d(result, input)
        # just Hook law like a spring where mu is the elasticity modulus
        result[1] = mu*input[1]
    end   
    action = FunctionAction(tensor_apply_1d, 1, 1)
    return AbstractBilinearForm("Hookian1D",gradient_operator, gradient_operator, action; regions = regions)
end

"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (C eps(u), eps(v)) where C is the 3D stiffness tensor
    for isotropic media in Voigt notation, i.e.
    C eps(u) = 2 mu eps(u) + lambda tr(eps(u)) for Lame parameters mu and lambda
    
    In Voigt notation C is a 3 x 3 matrix
    C = [c11,c12,  0
         c12,c11,  0
           0,  0,c33]
    
    where c33 = shear_modulus, c12 = lambda and c11 = c33 + c12
    
"""
function HookStiffnessOperator2D(mu::Real, lambda::Real; regions::Array{Int,1} = [0], gradient_operator = SymmetricGradient)
    function tensor_apply_2d(result, input)
        result[1] = (lambda + 2*mu)*input[1] + lambda*input[2]
        result[2] = (lambda + 2*mu)*input[2] + lambda*input[1]
        result[3] = mu*input[3]
    end   
    action = FunctionAction(tensor_apply_2d, 3, 2)
    return AbstractBilinearForm("Hookian2D",gradient_operator, gradient_operator, action; regions = regions)
end

"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (C eps(u), eps(v)) where C is the 3D stiffness tensor
for isotropic media in Voigt notation, i.e.
C eps(u) = 2 mu eps(u) + lambda tr(eps(u)) for Lame parameters mu and lambda

In Voigt notation C is a 6 x 6 matrix
C = [c11,c12,c12,  0,  0,  0
     c12,c11,c12,  0,  0,  0
     c12,c12,c11,  0,  0,  0
       0,  0,  0,c44,  0,  0
       0,  0,  0,  0,c44,  0
       0,  0,  0,  0,  0,c44]

where c44 = shear_modulus, c12 = lambda and c11 = c44 + c12
    
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
    action = FunctionAction(tensor_apply_3d, 6, 3)
    return AbstractBilinearForm("Hookian3D", gradient_operator, gradient_operator, action; regions = regions)
end


"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (A(u),v) or (u,A(v)) with some user-specified action A
    
"""
function ReactionOperator(action::AbstractAction; apply_action_to = 1, identity_operator = Identity, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("Reaction",identity_operator, identity_operator, action; apply_action_to = apply_action_to, regions = regions)
end

"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (beta*grad(u),v) with some user-specified function beta with the
interface beta(result,x) (so it writes its result into result and returns nothing)
    
"""
function ConvectionOperator(beta::Function, xdim::Int, ncomponents::Int; bonus_quadorder::Int = 0, testfunction_operator::Type{<:AbstractFunctionOperator} = Identity, regions::Array{Int,1} = [0])
    function convection_function_func() # dot(convection!, input=Gradient)
        convection_vector = zeros(Float64,xdim)
        function closure(result, input, x)
            # evaluate beta
            beta(convection_vector,x)
            # compute (beta*grad)u
            for j = 1 : ncomponents
                result[j] = 0.0
                for k = 1 : xdim
                    result[j] += convection_vector[k]*input[(j-1)*xdim+k]
                end
            end
        end    
    end    
    convection_action = XFunctionAction(convection_function_func(), ncomponents, xdim; bonus_quadorder = bonus_quadorder)
    return AbstractBilinearForm("(a(=XFunction) * Gradient) u * v", Gradient,testfunction_operator, convection_action; regions = regions, transposed_assembly = true)
end


"""
$(TYPEDEF)

abstract multi-linearform of the form

m(v1,v2,...,vk) = (A(O(v1),O(v2),...,O(vk-1)),Ok(vk))

(so far only intended for use as RHSOperator with MLFeval)
"""
mutable struct AbstractMultilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operators::Array{DataType,1}
    action::AbstractAction
    regions::Array{Int,1}
end


"""
$(TYPEDEF)

abstract nonlinearform operator 

can only be applied in PDE LHS
"""
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

function generate_newton_action_from_nlaction(nlaction::Function, size)
    return closure
end

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

    if ADnewton
        name = name * " [AD-Newton]"
        # the action for the LHS is calculated by automatic differentiation (AD)
        # from the given action_kernel
        result_temp = Vector{Float64}(undef,argsizes[1])
        input_temp = Vector{Float64}(undef,argsizes[2])
        jac_temp = Matrix{Float64}(undef,argsizes[1],argsizes[2])
        Dresult = DiffResults.DiffResult(result_temp,jac_temp)
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        cfg =ForwardDiff.JacobianConfig(action_kernel, result_temp, input_temp)
        function newton_kernel(result, input_current, input_ansatz)
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
        action = FunctionAction(newton_kernel, argsizes[1], dim)

        # the action for the RHS just evaluates the action_kernel at input_current
        function rhs_kernel(result, input_current, input_ansatz)
            action_kernel(result, input_current)
            return nothing
        end
        action_rhs = FunctionAction(rhs_kernel, argsizes[1], dim)
    else
        action = FunctionAction(action_kernel, argsizes[1], dim)
        if action_kernel_rhs != nothing
            action_rhs = FunctionAction(action_kernel_rhs, argsizes[1], dim)
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
$(TYPEDEF)

abstract trilinearform operator that assembles
- c(a,u,v) = int_regions action(operator1(a) * operator2(u))*operator3(v)

where a is one of the other unknowns of the PDEsystem

can only be applied in PDE LHS
"""
mutable struct AbstractTrilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperatorLHS
    name::String
    operator1::Type{<:AbstractFunctionOperator} # operator for argument 1
    operator2::Type{<:AbstractFunctionOperator} # operator for argument 1
    operator3::Type{<:AbstractFunctionOperator} # operator for argument 1
    a_from::Int     # unknown id where fixed argument takes its values from
    a_to::Int       # position of fixed argument
    action::AbstractAction # is applied to argument 1 and 2, i.e input consists of operator1(a),operator2(u)
    regions::Array{Int,1}
    transposed_assembly::Bool
end
"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (beta*grad(u),v) where beta is the id of some unknown of the PDEDescription.
With fixed_argument = 2 beta and u can siwtch their places.

    
"""
function ConvectionOperator(a_from::Int, beta_operator, xdim::Int, ncomponents::Int; fixed_argument::Int = 1, testfunction_operator::Type{<:AbstractFunctionOperator} = Identity, regions::Array{Int,1} = [0])
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
    convection_action = FunctionAction(convection_function_fe(), ncomponents)
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

"""
$(TYPEDEF)

constructor for AbstractBilinearForm that describes a(u,v) = (beta x curl(u),v)
where beta is the id of some unknown vector field of the PDEDescription, u and v
are also vector-fields and x is the cross product (so far this is only implemented in 2D)
    
"""
function ConvectionRotationFormOperator(beta::Int, beta_operator, xdim::Int, ncomponents::Int; testfunction_operator::Type{<:AbstractFunctionOperator} = Identity, regions::Array{Int,1} = [0])
    # action input consists of two inputs
    # input[1:xdim] = operator1(a)
    # input[xdim+1:end] = curl(u)
    function rotationform_2d()
        function closure(result, input)
            result[1] = input[2] * input[3]
            result[2] = - input[1] * input[3]
        end    
    end    
    convection_action = FunctionAction(rotationform_2d(), ncomponents)
    return AbstractTrilinearForm{ON_CELLS}("(a(=unknown $beta) x Curl2D u ) * v",beta_operator,Curl2D,testfunction_operator,beta, 1, convection_action, regions, true)
end


"""
````
mutable struct RhsOperator{AT<:AbstractAssemblyType} <: AbstractPDEOperatorRHS
    rhsfunction::Function
    testfunction_operator::Type{<:AbstractFunctionOperator}
    timedependent::Bool
    regions::Array{Int,1}
    xdim:: Int
    ncomponents:: Int
    bonus_quadorder:: Int
    store_operator::Bool                    # should the vector of the operator be stored?
    storage::AbstractArray{Float64,1}       # vector can be stored here to allow for fast reassembly in iterative settings
end
````

right-hand side operator

can only be applied in PDE RHS
"""
mutable struct RhsOperator{AT<:AbstractAssemblyType} <: AbstractPDEOperatorRHS
    rhsfunction::Function
    testfunction_operator::Type{<:AbstractFunctionOperator}
    timedependent::Bool
    regions::Array{Int,1}
    xdim:: Int
    ncomponents:: Int
    bonus_quadorder:: Int
    store_operator::Bool               # should the matrix representation of the operator be stored?
    storage::AbstractArray{Float64,1}  # matrix can be stored here to allow for fast matmul operations in iterative settings
end

function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    regions::Array{Int,1},
    rhsfunction!::Function,
    xdim::Int,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    on_boundary::Bool = false)

    # check if function is time-dependent
    if applicable(rhsfunction!,[0],0,0)
        timedependent = true
        rhsfunc = rhsfunction!
    else
        timedependent = false
        rhsfunc(result,x,t) = rhsfunction!(result,x)
    end

    if on_boundary == true
        return RhsOperator{ON_BFACES}(rhsfunc, operator, timedependent, regions, xdim, ncomponents, bonus_quadorder, false, [])
    else
        return RhsOperator{ON_CELLS}(rhsfunc, operator, timedependent, regions, xdim, ncomponents, bonus_quadorder, false, [])
    end
end


"""
$(TYPEDEF)

evaluation of a bilinearform where the second argument is fixed by given FEVectorBlock

can only be applied in PDE RHS
"""
struct BLFeval <: AbstractPDEOperatorRHS
    BLF::AbstractBilinearForm
    Data::FEVectorBlock
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end

function BLFeval(BLF, Data, factor; nonlinear::Bool = false, timedependent::Bool = false)
    return BLFeval(BLF, Data, factor, nonlinear, timedependent)
end

"""
$(TYPEDEF)

evaluation of a trilinearform where the first and  second argument is fixed by given FEVectorBlocks

can only be applied in PDE RHS
"""
struct TLFeval <: AbstractPDEOperatorRHS
    TLF::AbstractTrilinearForm
    Data1::FEVectorBlock
    Data2::FEVectorBlock
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end

function TLFeval(TLF, Data1, Data2, factor::Real = 1; nonlinear::Bool = false, timedependent::Bool = false)
    return TLFeval(TLF, Data1, Data2, factor, nonlinear, timedependent)
end

"""
$(TYPEDEF)

evaluation of a multi-linearform where the al but the last argument are fixed by given FEVectorBlocks

can only be applied in PDE RHS
"""
struct MLFeval <: AbstractPDEOperatorRHS
    MLF::AbstractMultilinearForm
    Data::Array{FEVectorBlock,1}
    factor::Real
    nonlinear::Bool
    timedependent::Bool
end

function MLFeval(MLF, Data, factor; nonlinear::Bool = false, timedependent::Bool = false)
    return MLFeval(MLF, Data, factor, nonlinear, timedependent)
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
    return false, O.timedependent
end


################ ASSEMBLY SPECIFICATIONS ################



# check if operator causes nonlinearity or time-dependence
function check_PDEoperator(O::AbstractPDEOperator)
    return false, false
end
function check_PDEoperator(O::AbstractTrilinearForm)
    return true, false
end
function check_PDEoperator(O::FVConvectionDiffusionOperator)
    return O.beta_from != 0, false
end
function check_PDEoperator(O::CopyOperator)
    return true, true
end
function check_PDEoperator(O::BLFeval)
    return O.nonlinear, O.timedependent
end
function check_PDEoperator(O::TLFeval)
    return O.nonlinear, O.timedependent
end
function check_PDEoperator(O::MLFeval)
    return O.nonlinear, O.timedependent
end
function check_PDEoperator(O::AbstractNonlinearForm)
    return true, false
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
    xCellDofs = FE1.dofmaps[CellDofs]
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
    evaluate!(O.fluxes,fluxIntegrator,CurrentSolution[c]; verbosity = verbosity - 1)

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
                flux *= 1 // 2
            end       
            if flux > 0
                A[cell,cell] += flux
                if other_cell > 0
                    A[other_cell,cell] -= flux
                end    
            else   
                if other_cell > 0
                    A[other_cell,other_cell] -= flux
                    A[cell,other_cell] += flux
                else
                    A[cell,cell] += flux
                end 
            end
        end
    end
end



function update_storage!(O::AbstractBilinearForm{AT}, CurrentSolution::FEVector, j::Int, k::Int; factor = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}

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

    # ensure that storage is large_enough
    FE = CurrentSolution[j].FES
    O.storage = zeros(Float64,FE.ndofs)

    function rhs_function() # result = F(v) = f*operator(v) = f*input
        temp = zeros(Float64,O.ncomponents)
        function closure(result,input,x)
            O.rhsfunction(temp,x, time)
            result[1] = 0
            for j = 1 : O.ncomponents
                result[1] += temp[j]*input[j] 
            end
        end
    end    
    action = XFunctionAction(rhs_function(),1,O.xdim; bonus_quadorder = O.bonus_quadorder)
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
        assemble!(b, CurrentSolution[fixed_component], BLF; apply_action_to = O.apply_action_to, factor = factor, verbosity = verbosity)
    end
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::TLFeval; factor = 1, time::Real = 0, verbosity::Int = 0)
    FE1 = O.Data1.FES
    FE2 = O.Data2.FES
    FE3 = b.FES
    TLF = TrilinearForm(Float64, ON_CELLS, FE1, FE2, FE3, O.TLF.operator1, O.TLF.operator2, O.TLF.operator3, O.TLF.action; regions = O.TLF.regions)  
    assemble!(b, O.Data1, O.Data2, TLF; factor = factor * O.factor, verbosity = verbosity)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::MLFeval; factor = 1, time::Real = 0, verbosity::Int = 0)
    FES = []
    for k = 1 : length(O.Data)
        push!(FES, O.Data[k].FES)
    end
    push!(FES, b.FES)
    FES = Array{FESpace,1}(FES)
    MLF = MultilinearForm(Float64, ON_CELLS, FES, O.MLF.operators, O.MLF.action; regions = O.MLF.regions)  
    assemble!(b, O.Data, MLF; factor = factor * O.factor, verbosity = verbosity)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::BLFeval; factor = 1, time::Real = 0, verbosity::Int = 0)
    if O.BLF.store_operator == true
        addblock_matmul!(b,O.BLF.storage,O.Data; factor = factor)
    else
        FE1 = b.FES
        FE2 = O.Data.FES
        if FE1 == FE2 && O.BLF.operator1 == O.BLF.operator2
            BLF = SymmetricBilinearForm(Float64, ON_CELLS, FE1, O.BLF.operator1, O.BLF.action; regions = O.BLF.regions)    
        else
            BLF = BilinearForm(Float64, ON_CELLS, FE1, FE2, O.BLF.operator1, O.BLF.operator2, O.BLF.action; regions = O.BLF.regions)    
        end
        assemble!(b, O.Data, BLF; apply_action_to = O.BLF.apply_action_to, factor = factor * O.factor, verbosity = verbosity)
    end
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractTrilinearForm; time::Real = 0, verbosity::Int = 0)
    FE1 = CurrentSolution[O.a_from].FES
    FE2 = A.FESX
    FE3 = A.FESY
    TLF = TrilinearForm(Float64, ON_CELLS, FE1, FE2, FE3, O.operator1, O.operator2, O.operator3, O.action; regions = O.regions)  
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
        function rhs_function() # result = F(v) = f*operator(v) = f*input
            temp = zeros(Float64,O.ncomponents)
            function closure(result,input,x)
                O.rhsfunction(temp,x,time)
                result[1] = 0
                for j = 1 : O.ncomponents
                    result[1] += temp[j]*input[j] 
                end
            end
        end    
        action = XFunctionAction(rhs_function(),1,O.xdim; bonus_quadorder = O.bonus_quadorder)
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
    assemble!(A, NLF, CurrentSolution[O.coeff_from]; verbosity = verbosity, transposed_assembly = O.transposed_assembly)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::AbstractNonlinearForm; time::Real = 0, verbosity::Int = 0)
    FE = Array{FESpace,1}(undef, length(O.coeff_from))
    for j = 1 : length(O.coeff_from)
        FE[j] = CurrentSolution[O.coeff_from[j]].FES
    end
    FE2 = b.FES
    NLF = NonlinearForm(Float64, ON_CELLS, FE, FE2, O.operator1, O.operator2, O.action_rhs; regions = O.regions)  
    assemble!(b, NLF, CurrentSolution[O.coeff_from]; verbosity = verbosity)
end
