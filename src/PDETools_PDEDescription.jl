######################
# AbstactPDEOperator #
######################
#
# to describe operators in the (weak form of the) PDE
#
# some intermediate layer that knows nothing of the FE discretisatons
# but triggers certain AssemblyPatterns/AbstractActions when called for assembly!
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


abstract type AbstractBoundaryType end
abstract type DirichletBoundary <: AbstractBoundaryType end
abstract type BestapproxDirichletBoundary <: DirichletBoundary end
abstract type InterpolateDirichletBoundary <: DirichletBoundary end
abstract type HomogeneousDirichletBoundary <: DirichletBoundary end


########################
### BoundaryOperator ###
########################
#
# collects boundary data for a component of the system and allows to specify a AbstractBoundaryType for each boundary region
# so far only DirichletBoundary types (see above)
#
# later also SymmetryBoundary
#
# Note: NeumannBoundary has to be implemented as a RhsOperator with on_boundary = true
# Note: PeriodicBoundary has to be implemented as a CombineDofs <: AbstractGlobalConstraint
#
struct BoundaryOperator <: AbstractPDEOperator
    regions4boundarytype :: Dict{Type{<:AbstractBoundaryType},Array{Int,1}}
    data4bregion :: Array{Any,1}
    quadorder4bregion :: Array{Int,1}
    xdim :: Int
    ncomponents :: Int
end

function BoundaryOperator(xdim::Int, ncomponents::Int = 1)
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = zeros(Int,0)
    return BoundaryOperator(regions4boundarytype, [], quadorder4bregion, xdim, ncomponents)
end

function BoundaryOperator(boundarytype4bregion::Array{DataType,1}, data4region, xdim::Int, ncomponents::Int = 1; bonus_quadorder::Int = 0)
    regions4boundarytype = Dict{Type{<:AbstractBoundaryType},Array{Int,1}}()
    quadorder4bregion = ones(Int,length(boundarytype4bregion))*bonus_quadorder
    for j = 1 : length(boundarytype4bregion)
        btype = boundarytype4bregion[j]
        regions4boundarytype[btype]=push!(get(regions4boundarytype, btype, []),j)
        
    end
    return BoundaryOperator(regions4boundarytype, data4region, quadorder4bregion, xdim, ncomponents)
end

function Base.append!(O::BoundaryOperator,region::Int, btype::Type{<:AbstractBoundaryType}; data = Nothing, bonus_quadorder::Int = 0)
    O.regions4boundarytype[btype]=push!(get(O.regions4boundarytype, btype, []),region)
    while length(O.data4bregion) < region
        push!(O.data4bregion, Nothing)
    end
    while length(O.quadorder4bregion) < region
        push!(O.quadorder4bregion, 0)
    end
    O.quadorder4bregion[region] = bonus_quadorder
    O.data4bregion[region] = data
end


function Base.append!(O::BoundaryOperator,regions::Array{Int,1}, btype::Type{<:AbstractBoundaryType}; data = Nothing, bonus_quadorder::Int = 0)
    for j = 1 : length(regions)
        append!(O,regions[j], btype; data = data, bonus_quadorder = bonus_quadorder)
    end
end


# type to steer when a PDE block is (re)assembled
abstract type AbstractAssemblyTrigger end
abstract type AssemblyInitial <: AbstractAssemblyTrigger end    # is only assembled in initial assembly
abstract type AssemblyEachTimeStep <: AssemblyInitial end       # is (re)assembled in each timestep
abstract type AssemblyAlways <: AssemblyEachTimeStep end        # is always (re)assembled
abstract type AssemblyFinal <: AbstractAssemblyTrigger end       # is only assembled after solving




#################################
### AbstractGlobalConstraints ###
#################################
#
# further constraints that cannot be described with (sparse) PDEOperators and are realized by manipulations of the
# already assembled system
#
# FixedIntegralMean: Ensure that integral mean of a _component_ attains _value_
# (e.g. for pressure in Stokes, avoids full row/column in matrix)
#
# CombineDofs: Identify given dofs of two components with each other
# (might be used for periodic boundary conditions, or using different FETypes in different regions)
#
# NOTE: constraints like zero divergence have to be realised with PDEOperators like LagrangeMultiplier
#
abstract type AbstractGlobalConstraint end

struct FixedIntegralMean <: AbstractGlobalConstraint
    name::String
    component::Int
    value::Real
    when_assemble::Type{<:AbstractAssemblyTrigger}
end 
function FixedIntegralMean(component::Int, value::Real)
    return FixedIntegralMean("Mean[$component] != $value",component, value, AssemblyFinal)
end

struct CombineDofs <: AbstractGlobalConstraint
    name::String
    componentX::Int                  # component nr for dofsX
    componentY::Int                  # component nr for dofsY
    dofsX::Array{Int,1}     # dofsX that should be the same as dofsY in Y component
    dofsY::Array{Int,1}
    when_assemble::Type{<:AbstractAssemblyTrigger}
end 
function CombineDofs(componentX,componentY,dofsX,dofsY)
    @assert length(dofsX) == length(dofsY)
    return CombineDofs("CombineDofs[$componentX,$componentY] (ndofs = $(length(dofsX)))",componentX,componentY,dofsX,dofsY, AssemblyAlways)
end


##################
# PDEDescription #
##################
#
# A PDE is described by an nxn matrix and vector of PDEOperator
# the indices of matrix relate to FEBlocks given to solver
# all Operators of [i,k] - matrixblock are assembled into system matrix block [i*n+k]
# all Operators of [i] - rhs-block are assembled into rhs block [i]
#
# additionally BoundayOperators and GlobalConstraints are assigned to handle
# boundary data and global side constraints (like a fixed global integral mean)

mutable struct PDEDescription
    name::String
    LHSOperators::Array{Array{AbstractPDEOperator,1},2}
    RHSOperators::Array{Array{AbstractPDEOperator,1},1}
    BoundaryOperators::Array{BoundaryOperator,1}
    GlobalConstraints::Array{AbstractGlobalConstraint,1}
end

function PDEDescription(name, LHS, RHS, BoundaryOperators)
    nFEs = length(RHS)
    NoConstraints = Array{AbstractGlobalConstraint,1}(undef,0)
    return PDEDescription(name, LHS, RHS, BoundaryOperators, NoConstraints)
end



function Base.show(io::IO, PDE::PDEDescription)
    println("\nPDE-DESCRIPTION")
    println("===============")
    println("  name = $(PDE.name)\n")

    println("  LHS block | PDEOperator(s)")
    for j=1:size(PDE.LHSOperators,1), k=1:size(PDE.LHSOperators,2)
        if length(PDE.LHSOperators[j,k]) > 0
            print("    [$j,$k]   | ")
            for o = 1 : length(PDE.LHSOperators[j,k])
                try
                    print("$(PDE.LHSOperators[j,k][o].name) (regions = $(PDE.LHSOperators[j,k][o].regions))")
                catch
                    print("$(PDE.LHSOperators[j,k][o].name) (regions = [0])")
                end
                if o == length(PDE.LHSOperators[j,k])
                    println("")
                else
                    print("\n            | ")
                end
            end
        else    
            println("    [$j,$k]   | none")
        end
    end

    println("\n  RHS block | PDEOperator(s)")
    for j=1:size(PDE.RHSOperators,1)
        if length(PDE.RHSOperators[j]) > 0
            print("     [$j]    | ")
            for o = 1 : length(PDE.RHSOperators[j])
                print("$(typeof(PDE.RHSOperators[j][o])) (regions = $(PDE.RHSOperators[j][o].regions))")
                if o == length(PDE.RHSOperators[j])
                    println("")
                else
                    print("\n            | ")
                end
            end
        else    
            println("     [$j]    | none")
        end
    end

    println("")
    for j=1:length(PDE.BoundaryOperators)
        print("   BoundaryOperator[$j] : ")
        try
            if length(PDE.BoundaryOperators[j].regions4boundarytype[BestapproxDirichletBoundary]) > 0
                print("BestapproxDirichletBoundary -> $(PDE.BoundaryOperators[j].regions4boundarytype[BestapproxDirichletBoundary])\n                         ")
            end
        catch
        end
        try
            if length(PDE.BoundaryOperators[j].regions4boundarytype[InterpolateDirichletBoundary]) > 0
                print("InterpolateDirichletBoundary -> $(PDE.BoundaryOperators[j].regions4boundarytype[InterpolateDirichletBoundary])\n                         ")
            end
        catch
        end
        try
            if length(PDE.BoundaryOperators[j].regions4boundarytype[HomogeneousDirichletBoundary]) > 0
                print("HomogeneousDirichletBoundary -> $(PDE.BoundaryOperators[j].regions4boundarytype[HomogeneousDirichletBoundary])\n                          ")
            end
        catch
        end
        println("")
    end

    println("")
    for j=1:length(PDE.GlobalConstraints)
        println("  GlobalConstraints[$j] : $(PDE.GlobalConstraints[j].name)")
    end
end
