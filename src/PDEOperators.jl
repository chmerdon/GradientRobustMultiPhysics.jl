
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
function DiagonalOperator(value::Real = 1.0, onlynz::Bool = true; regions::Array{Int,1} = [0])
    return DiagonalOperator("Diag($value)",value, onlynz, regions)
end


####################
### CopyOperator ###
####################
#
# copies entries from TargetVector to rhs block
#
struct CopyOperator <: AbstractPDEOperator
    name::String
    copy_from::Int
    factor::Real
end
function CopyOperator(copy_from, factor)
    return CopyOperator("CopyOperator",copy_from, factor)
end

############################
### AbstractBilinearForm ###
############################
#
# expects two operators _operator1_ and _operator2_ and an _action_ and an AT::AbtractAssemblyType and _regions_
# 
# and assembles b(u,v) = int_regions action(operator1(u)) * operator2(v) if apply_action_to = 1
#            or b(u,v) = int_regions operator1(u) * action(operator2(v)) if apply_action_to = 2
#
mutable struct AbstractBilinearForm{AT<:AbstractAssemblyType} <: AbstractPDEOperator
    name::String
    operator1::Type{<:AbstractFunctionOperator}
    operator2::Type{<:AbstractFunctionOperator}
    action::AbstractAction
    apply_action_to::Int
    regions::Array{Int,1}
    store_operator::Bool                    # should the matrix repsentation of the operator be stored?
    storage::AbstractArray{Float64,2}  # matrix can be stored here to allow for fast matmul operations in iterative settings
end
function AbstractBilinearForm(name, operator1,operator2, action; apply_action_to = 1, regions::Array{Int,1} = [0])
    return AbstractBilinearForm{AbstractAssemblyTypeCELL}(name,operator1, operator2, action, apply_action_to, regions,false,zeros(Float64,0,0))
end
function AbstractBilinearForm(operator1,operator2; apply_action_to = 1, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("$operator1 x $operator2",operator1, operator2, DoNotChangeAction(1); apply_action_to = apply_action_to, regions = regions)
end
function LaplaceOperator(diffusion::Real = 1.0, xdim::Int = 2, ncomponents::Int = 1; gradient_operator = Gradient, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("Laplacian",gradient_operator, gradient_operator, MultiplyScalarAction(diffusion, ncomponents*xdim); regions = regions)
end
# todo
# here a general connection to arbitrary tensors C_ijkl (encencodedoded as an action) is possible in future
function HookStiffnessOperator1D(mu::Real; regions::Array{Int,1} = [0], gradient_operator = TangentialGradient)
    function tensor_apply_1d(result, input)
        # just Hook law like a spring where mu is the elasticity modulus
        result[1] = mu*input[1]
    end   
    action = FunctionAction(tensor_apply_1d, 1, 1)
    return AbstractBilinearForm("Hookian1D",gradient_operator, gradient_operator, action; regions = regions)
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
    return AbstractBilinearForm("Hookian2D",gradient_operator, gradient_operator, action; regions = regions)
end
function ReactionOperator(action::AbstractAction; apply_action_to = 1, identity_operator = Identity, regions::Array{Int,1} = [0])
    return AbstractBilinearForm("Reaction",identity_operator, identity_operator, action; apply_action_to = apply_action_to, regions = regions)
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


struct BLFeval <: AbstractPDEOperator
    BLF::AbstractBilinearForm
    Data::FEVectorBlock
end


##################################
### FVUpwindDivergenceOperator ###
##################################
#
# finite-volume upwind divergence div_upw(beta*rho)
#
# assumes rho is constant on each cell
# 
# (1) calculate normalfluxes from component at _beta_from_
# (2) compute upwind divergence on each cell and put coefficient in matrix
#           div_upw(beta*rho)|_T = sum_{F face of T} normalflux(F) * rho(F)
#
#           where rho(F) is the rho in upwind direction 
#
#     and put it into P0xP0 matrix block like this:
#
#           Loop over cell, face of cell
#
#               other_cell = other face neighbour cell
#               if flux := normalflux(F_j) * CellSigns[face,cell] > 0
#                   A(cell,cell) += flux
#                   A(other_cell,cell) -= flux
#               else
#                   A(other_cell,other_cell) -= flux
#                   A(cell,other_cell) += flux
#                   
# see coressponding assemble! routine

mutable struct FVUpwindDivergenceOperator <: AbstractPDEOperator
    name::String
    beta_from::Int                   # component that determines
    fluxes::Array{Float64,2}         # saves normalfluxes of beta here
end
function FVUpwindDivergenceOperator(beta_from::Int)
    @assert beta_from > 0
    fluxes = zeros(Float64,0,1)
    return FVUpwindDivergenceOperator("FVUpwindDivergence",beta_from,fluxes)
end


################ ASSEMBLY SPECIFICATIONS ################



# check if operator causes nonlinearity or time-dependence
function check_PDEoperator(O::AbstractPDEOperator)
    return false, false
end
function check_PDEoperator(O::ConvectionOperator)
    return O.beta_from != 0, false
end
function check_PDEoperator(O::FVUpwindDivergenceOperator)
    return O.beta_from != 0, false
end
function check_PDEoperator(O::CopyOperator)
    return true, true
end

function check_dependency(O::AbstractPDEOperator, arg::Int)
    return false
end

function check_dependency(O::Union{ConvectionOperator,FVUpwindDivergenceOperator}, arg::Int)
    return O.beta_from == arg
end



function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::DiagonalOperator; time::Real = 0, verbosity::Int = 0)
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


function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::FVUpwindDivergenceOperator; time::Real = 0, verbosity::Int = 0)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert FE1 == FE2
    xFaceNodes = FE1.xgrid[FaceNodes]
    xFaceNormals = FE1.xgrid[FaceNormals]
    xFaceCells = FE1.xgrid[FaceCells]
    xFaceVolumes = FE1.xgrid[FaceVolumes]
    xCellFaces = FE1.xgrid[CellFaces]
    xCellSigns = FE1.xgrid[CellSigns]
    nfaces = num_sources(xFaceNodes)
    ncells = num_sources(xCellSigns)
    nnodes = num_sources(FE1.xgrid[Coordinates])
    
    # ensure that flux field is long enough
    if length(O.fluxes) < nfaces
        O.fluxes = zeros(Float64,nfaces,1)
    end

    # compute normal fluxes of component beta
    c = O.beta_from
    fill!(O.fluxes,0)
    fluxIntegrator = ItemIntegrator{Float64,AbstractAssemblyTypeFACE}(NormalFlux, DoNotChangeAction(1), [0])
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
            flux = - O.fluxes[face] * xCellSigns[cf,cell]
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
                end 
            end
        end
    end
end



function update_storage!(O::AbstractBilinearForm{AT}, CurrentSolution::FEVector, j::Int, k::Int; factor::Real = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}

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

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::AbstractBilinearForm{AT}; factor::Real = 1, time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    if O.store_operator == true
        addblock(A,O.storage; factor = factor)
    else
        FE1 = A.FESX
        FE2 = A.FESY
        if FE1 == FE2 && O.operator1 == O.operator2
            BLF = SymmetricBilinearForm(Float64, AT, FE1, O.operator1, O.action; regions = O.regions)    
        else
            BLF = BilinearForm(Float64, AT, FE1, FE2, O.operator1, O.operator2, O.action; regions = O.regions)    
        end
        assemble!(A, BLF; apply_action_to = O.apply_action_to, factor = factor, verbosity = verbosity)
    end
end


function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::AbstractBilinearForm{AT}; factor::Real = 1, time::Real = 0, verbosity::Int = 0, fixed_component::Int = 0) where {AT<:AbstractAssemblyType}
    if O.store_operator == true
        addblock_matmul(b,O.storage,CurrentSolution[fixed_component]; factor = factor)
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


function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::BLFeval; factor::Real = 1, time::Real = 0, verbosity::Int = 0, fixed_component::Int = 0)
    if O.BLF.store_operator == true
        addblock_matmul(b,O.BLF.storage,O.Data; factor = factor)
    else
        FE1 = b.FES
        FE2 = O.Data.FES
        if FE1 == FE2 && O.BLF.operator1 == O.BLF.operator2
            BLF = SymmetricBilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, O.BLF.operator1, O.BLF.action; regions = O.BLF.regions)    
        else
            BLF = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, O.BLF.operator1, O.BLF.operator2, O.BLF.action; regions = O.BLF.regions)    
        end
        assemble!(b, O.Data, BLF; apply_action_to = O.BLF.apply_action_to, factor = factor, verbosity = verbosity)
    end
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::ConvectionOperator; time::Real = 0, verbosity::Int = 0)
    if O.beta_from == 0
        FE1 = A.FESX
        FE2 = A.FESY
        ConvectionForm = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, Gradient, O.testfunction_operator, O.action; regions = O.regions)  
        assemble!(A, ConvectionForm; verbosity = verbosity, transposed_assembly = true)
    else
        FE1 = A.FESX
        FE2 = A.FESY
        ConvectionForm = TrilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE1, FE2, O.testfunction_operator, Gradient, O.testfunction_operator, O.action; regions = O.regions)  
        assemble!(A, ConvectionForm, CurrentSolution[O.beta_from]; verbosity = verbosity, transposed_assembly = true)
    end
end

function assemble!(A::FEMatrixBlock, CurrentSolution::FEVector, O::LagrangeMultiplier; time::Real = 0, verbosity::Int = 0, At::FEMatrixBlock)
    FE1 = A.FESX
    FE2 = A.FESY
    @assert At.FESX == FE2
    @assert At.FESY == FE1
    DivPressure = BilinearForm(Float64, AbstractAssemblyTypeCELL, FE1, FE2, O.operator, Identity, MultiplyScalarAction(-1.0,1))   
    assemble!(A, DivPressure; verbosity = verbosity, transpose_copy = At)
end

function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::RhsOperator{AT}; time::Real = 0, verbosity::Int = 0) where {AT<:AbstractAssemblyType}
    FE = b.FES
    RHS = LinearForm(Float64,AT, FE, O.testfunction_operator, O.action; regions = O.regions)
    assemble!(b, RHS; verbosity = verbosity)
end


function assemble!(b::FEVectorBlock, CurrentSolution::FEVector, O::CopyOperator; time::Real = 0, verbosity::Int = 0) 
    for j = 1 : length(b)
        b[j] = CurrentSolution[O.copy_from][j] * O.factor
    end
end
