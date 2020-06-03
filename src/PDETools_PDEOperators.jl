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
#       assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractPDEOperator) # if intended to use in LHS
#       assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::AbstractPDEOperator) # if intended to use in RHS
#
#   (2) to allow SolverConfig to check if operator is nonlinear, timedependent:
#       Bool, Bool = check_PDEoperator(O::AbstractPDEOperator)
# 


abstract type AbstractPDEOperator end
abstract type NoConnection <: AbstractPDEOperator end # => empy block in matrix

########################
### DiagonalOperator ###
########################
#
# puts _value_ on the diagonal entries of the cell dofs within given _regions_
# if _onlyz_ == true only values that are zero are changed
#
struct DiagonalOperator <: AbstractPDEOperator
    name::String
    value::Real
    onlyz::Bool
    regions::Array{Int,1}
end
function DiagonalOperator(value::Real = 1.0, onlynz::Bool = true; regions::Array{Int,1} = [])
    return DiagonalOperator("Diag($value)",value, onlynz, regions)
end


############################
### AbstractBilinearForm ###
############################
#
# expects two operators _operator1_ and _operator2_ and an _action_ and an AT::AbtractAssemblyType and _regions_
# 
# and assembles b(u,v) = int_regions action(operator1(u)) * operator2(v)
#
struct AbstractBilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperator
    name::String
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    regions::Array{Int,1}
end
function AbstractBilinearForm(operator1,operator2;regions::Array{Int,1} = [0])
    return AbstractBilinearForm{AbstractAssemblyTypeCELL}("$operator1 x $operator2",operator1, operator2, DoNotChangeAction(1), regions)
end
function LaplaceOperator(diffusion::Real = 1.0, xdim::Int = 2, ncomponents::Int = 1; gradient_operator = Gradient, regions::Array{Int,1} = [0])
    return AbstractBilinearForm{AbstractAssemblyTypeCELL}("Laplacian",gradient_operator, gradient_operator, MultiplyScalarAction(diffusion, ncomponents*xdim), regions)
end
# todo
# here a general connection to arbitrary tensors C_ijkl (encencodedoded as an action) is possible in future
function HookStiffnessOperator1D(mu::Real; regions::Array{Int,1} = [0], gradient_operator = TangentialGradient)
    function tensor_apply_1d(result, input)
        # just Hook law like a spring where mu is the elasticity modulus
        result[1] = mu*input[1]
    end   
    action = FunctionAction(tensor_apply_1d, 1, 1)
    return AbstractBilinearForm{AbstractAssemblyTypeCELL}("Hookian1D",gradient_operator, gradient_operator, action, regions)
end
function HookStiffnessOperator2D(mu::Real, lambda::Real; regions::Array{Int,1} = [0], gradient_operator = SymmetricGradient)
    function tensor_apply_2d(result, input)
        # compute sigma_ij = C_ijkl eps_kl
        # where input = [eps_11,eps_12,eps_21] is the symmetric gradient in Voigt notation
        # and result = [sigma_11,sigma_12,sigma_21] is Voigt representation of sigma_11
        # the tensor C is just a 3x3 matrix
        result[1] = (lambda + 2*mu)*input[1] + lambda*input[2]
        result[2] = (lambda + 2*mu)*input[2] + lambda*input[1]
        result[3] = mu*input[3]
    end   
    action = FunctionAction(tensor_apply_2d, 3, 2)
    return AbstractBilinearForm{AbstractAssemblyTypeCELL}("Hookian2D",gradient_operator, gradient_operator, action, regions)
end
function ReactionOperator(action::AbstractAction; identity_operator = Identity, regions::Array{Int,1} = [0])
    return AbstractBilinearForm{AbstractAssemblyTypeCELL}("Reaction",identity_operator, identity_operator, action, regions)
end



struct LagrangeMultiplier <: AbstractPDEOperator
    name::String
    operator::Type{<:AbstractFunctionOperator} # e.g. Divergence, automatically aligns with transposed block
end
function LagrangeMultiplier(operator::Type{<:AbstractFunctionOperator})
    return LagrangeMultiplier("LagrangeMultiplier($operator)",operator)
end

struct ConvectionOperator <: AbstractPDEOperator
    name::String
    action::AbstractAction                                      #      ----ACTION-----
    beta_from::Int                                              # e.g. (beta * grad) u * testfunction_operator(v)
    testfunction_operator::Type{<:AbstractFunctionOperator}     # beta_from = 0 if beta is encoded in action
                                                                # =beta_from  k if beta is from k-th PDE unknown (=> fixpoint iteration)
    regions::Array{Int,1}
end

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
    return ConvectionOperator("Convection(XFunction)",convection_action,0,testfunction_operator, regions)
end

function ConvectionOperator(beta::Int, xdim::Int, ncomponents::Int; testfunction_operator::Type{<:AbstractFunctionOperator} = Identity, regions::Array{Int,1} = [0])
    # action input consists of two inputs
    # input[1:ncomponents] = operator1(beta)
    # input[ncomponents+1:length(input)] = u
    function convection_function_fe()
        function closure(result, input)
            for j = 1 : ncomponents
                result[j] = 0.0
                for k = 1 : xdim
                    result[j] += input[k]*input[ncomponents+(j-1)*xdim+k]
                end
            end
        end    
    end    
    convection_action = FunctionAction(convection_function_fe(), ncomponents)
    return ConvectionOperator("Convection(Component[$beta])",convection_action,beta,testfunction_operator, regions)
end

struct RhsOperator{AT<:AbstractAssemblyType} <: AbstractPDEOperator
    action::AbstractAction                                  #       -----ACTION----
    testfunction_operator::Type{<:AbstractFunctionOperator} # e.g.  f * testfunction_operator(v)
    regions::Array{Int,1}
end

function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    data4region,
    xdim::Int,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    regions::Array{Int,1} = [0],
    on_boundary::Bool = false)

    function rhs_function() # result = F(v) = f*operator(v) = f*input
        temp = zeros(Float64,ncomponents)
        function closure(result,input,x,region)
            data4region[region](temp,x)
            result[1] = 0
            for j = 1 : ncomponents
                result[1] += temp[j]*input[j] 
            end
        end
    end    

    action = RegionWiseXFunctionAction(rhs_function(),1,xdim; bonus_quadorder = bonus_quadorder)
    if on_boundary == true
        return RhsOperator{AbstractAssemblyTypeBFACE}(action, operator, regions)
    else
        return RhsOperator{AbstractAssemblyTypeCELL}(action, operator, regions)
    end
end


function RhsOperator(
    operator::Type{<:AbstractFunctionOperator},
    data4allregions::Function,
    xdim::Int,
    ncomponents::Int = 1;
    bonus_quadorder::Int = 0,
    regions::Array{Int,1} = [0],
    on_boundary::Bool = false)

    function rhs_function() # result = F(v) = f*operator(v) = f*input
        temp = zeros(Float64,ncomponents)
        function closure(result,input,x)
            data4allregions(temp,x)
            result[1] = 0
            for j = 1 : ncomponents
                result[1] += temp[j]*input[j] 
            end
        end
    end    
    action = XFunctionAction(rhs_function(),1,xdim; bonus_quadorder = bonus_quadorder)
    if on_boundary == true
        return RhsOperator{AbstractAssemblyTypeBFACE}(action, operator, regions)
    else
        return RhsOperator{AbstractAssemblyTypeCELL}(action, operator, regions)
    end
end



################ ASSEMBLY SPECIFICATIONS ################



# check if operator causes nonlinearity or time-dependence
function check_PDEoperator(O::AbstractPDEOperator)
    return false, false
end
function check_PDEoperator(O::ConvectionOperator)
    return O.beta_from != 0, false
end



function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::DiagonalOperator; verbosity::Int = 0)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xCellDofs = FE1.CellDofs
    xCellRegions = FE1.xgrid[CellRegions]
    ncells = num_sources(xCellDofs)
    dof::Int = 0
    for item = 1 : ncells
        for r = 1 : length(O.regions)
            # check if item region is in regions
            if xCellRegions[item] == O.regions[r]
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


function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractBilinearForm{AT}; verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    FE1 = A.FESX
    FE2 = A.FESY
    if FE1 == FE2 && O.operator1 == O.operator2
        BLF = SymmetricBilinearForm(Float64, AT, FE1, O.operator1, O.action; regions = O.regions)    
    else
        BLF = BilinearForm(Float64, AT, FE1, FE2, O.operator1, O.operator2, O.action; regions = O.regions)    
    end
    FEAssembly.assemble!(A, BLF; verbosity = verbosity)
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::ConvectionOperator; verbosity::Int = 0)
    if O.beta_from == 0
        FE1 = A.FESX
        FE2 = A.FESY
        ConvectionForm = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, Gradient, O.testfunction_operator, O.action; regions = O.regions)  
        FEAssembly.assemble!(A, ConvectionForm; verbosity = verbosity)
    else
        FE1 = A.FESX
        FE2 = A.FESY
        ConvectionForm = TrilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE1, FE2, O.testfunction_operator, Gradient, O.testfunction_operator, O.action; regions = O.regions)  
        FEAssembly.assemble!(A, ConvectionForm, CurrentSolution[O.beta_from]; verbosity = verbosity)
    end
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::LagrangeMultiplier; verbosity::Int = 0, At::FEMatrixBlock)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert At.FESX == FE2
    @assert At.FESY == FE1
    DivPressure = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, O.operator, Identity, MultiplyScalarAction(-1.0,1))   
    FEAssembly.assemble!(At, DivPressure; verbosity = verbosity, transpose_copy = A)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::RhsOperator{AT}; verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    FE = b.FES
    RHS = LinearForm(Float64,AT, FE, O.testfunction_operator, O.action; regions = O.regions)
    FEAssembly.assemble!(b, RHS; verbosity = verbosity)
end
