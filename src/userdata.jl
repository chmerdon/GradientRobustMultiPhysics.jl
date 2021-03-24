abstract type AbstractUserDataType end
abstract type AbstractDataFunction <: AbstractUserDataType end
abstract type AbstractExtendedDataFunction <: AbstractDataFunction end
abstract type AbstractActionKernel <: AbstractUserDataType end
abstract type AbstractNLActionKernel <: AbstractActionKernel end

struct UserData{UST<: AbstractUserDataType,ndim}
    name::String
    dependencies::String
    quadorder::Int
    dimensions::SVector{ndim,Int}     # length of result and input arrays
    user_function::Function
    negotiated_function::Function
end

"""
````
function ActionKernel(
    f::Function,
    dimensions::Array{Int,1};
    name = "user action kernel",
    dependencies::String = "",
    quadorder::Int = 0)
````

Provides a negotation interface for some function that can be used in the Action constructor to define a user-defined operator action.
The function has to obey the interface

    f(result, input, [X, T, R, I, L])

where the parameters X (= space coordinates), T ( = time), R (= region number), I (= item number), L (= local coordinates) are optional.
Which of them are used has to be specified in the String dependencies. The string "XT" for example specifies that the interface is
only space and time dependent, i.e

    f(result, input, X, T)

The input vector usually provides the FunctionOperator evaluations of (a subset of) the ansatz arguments of the assembly pattern where the action is used.
The array dimensions specifies the expected length of result and input and quadorder determines the additional quadrature order to be used if this
function (or its derived action) is involved in an assembly process.
"""
function ActionKernel(f::Function, dimensions::Array{Int,1}; name = "user action kernel", dependencies::String = "", quadorder::Int = 0)

    nf = (result, input,X,T,R,I,L) -> f(result, input) # no other dependencies
    if dependencies == "X"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X)
    elseif dependencies == "T"
        nf = (result, input,X,T,R,I,L) -> f(result, input, T)
    elseif dependencies == "R"
        nf = (result, input,X,T,R,I,L) -> f(result, input, R)
    elseif dependencies == "I"
        nf = (result, input,X,T,R,I,L) -> f(result, input, I)
    elseif dependencies == "L"
        nf = (result, input,X,T,R,I,L) -> f(result, input, L)
    elseif dependencies == "XR"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, R)
    elseif dependencies == "XT"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, T)
    elseif dependencies == "XI"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, I)
    elseif dependencies == "XL"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, L)
    elseif dependencies == "TI"
        nf = (result, input,X,T,R,I,L) -> f(result, input, T, I)
    elseif dependencies == "RI"
        nf = (result, input,X,T,R,I,L) -> f(result, input, R, I)
    elseif dependencies == "XTR"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, T, R)
    elseif dependencies == "XTI"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, T, I)
    elseif dependencies == "XTL"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, T, L)
    elseif dependencies == "XRI"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, R, I)
    elseif dependencies == "XTRI"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, T, R, I)
    elseif dependencies == "XRIL"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, R, I, L)
    elseif dependencies == "XTRIL"
        nf = (result, input,X,T,R,I,L) -> f(result, input, X, T, R, I, L)
    end

    return UserData{AbstractActionKernel,length(dimensions)}(name, dependencies, quadorder, dimensions, f, nf)
end



"""
````
function NLActionKernel(
    f::Function,
    dimensions::Array{Int,1};
    name = "nonlinear user action kernel",
    dependencies::String = "",
    quadorder::Int = 0)
````

Provides a negotation interface for some function to be used as a nonlinear action kernel that can be used in the
NonlinearOperator constructor without automatic differentiation. The function f has to obey the interface

    f(result, input_current, input_ansatz)

and can be seen as a linearisation of a nonlinearity that can depend on the operator evaluation of the current iterate (input_current)
and, as usual, the operator evaluations of the ansatz function (input_ansatz).

No further dependencies are allowed currently. Note, that this is a work-in-progress feature.
"""
function NLActionKernel(f::Function, dimensions::Array{Int,1}; name = "user nonlinear action kernel", dependencies::String = "", quadorder::Int = 0)
    if length(dimensions) == 2
        push!(dimensions,dimensions[2])
    end
    nf = (result, input_current, input_ansatz, X,T,R,I,L) -> f(result, input_current, input_ansatz) # no other dependencies
    if dependencies != ""
        println("nonlinear action kernels that depend on extra variables are not yet supported")
    end

    return UserData{AbstractNLActionKernel,length(dimensions)}(name, dependencies, quadorder, dimensions, f, nf)
end


"""
````
function DataFunction(
    f::Function,
    dimensions::Array{Int,1};
    name = "user data function",
    dependencies::String = "",
    quadorder::Int = 0)
````

Provides a negotation interface for some user-defined function that 
can be used in integrate! and boundary or right-hand side data assignments.
The function f has to obey the interface

    f(result, [X, T])

where the parameters X (= space coordinates) and T ( = time) are optional. Which of them are used has to be specified in the 
String dependencies. The string "X" for example specifies that the interface is only space-dependent, i.e

    f(result, X)

The array dimensions specifies the expected length of result and input and quadorder determines the additional quadrature order to be used if this
function is involved in some quadrature-requireing procedure.
"""
function DataFunction(f::Function, dimensions::Array{Int,1}; name = "user data function", dependencies::String = "", quadorder::Int = 0)
    nf = (result,X,T) -> f(result) # no other dependencies
    if dependencies == "X"
        nf = (result,X,T) -> f(result, X)
    elseif dependencies == "T"
        nf = (result,X,T) -> f(result, T)
    elseif dependencies == "XT"
        nf = (result,X,T) -> f(result, X, T)
    end

    return UserData{AbstractDataFunction,length(dimensions)}(name, dependencies, quadorder, dimensions, f, nf)
end


function DataFunction(c::Array{<:Real,1}; name = "constant user data", quadorder::Int = 0)
    dimensions = [length(c),0]
    function f_from_c(result)
        result .= c
    end

    return DataFunction(f_from_c, dimensions; name = name, dependencies = "", quadorder = quadorder)
end

"""
````
function ExtendedDataFunction(
    f::Function,
    dimensions::Array{Int,1};
    name = "user data function",
    dependencies::String = "",
    quadorder::Int = 0)
````

Provides a negotation interface for some data function with extended dependencies
(region number, item number and local coordinates) that can be used in integrate!.
The function f has to obey the interface

    f(result, [X, T, R, I, L])

where the parameters X (= space coordinates) and T ( = time) are optional. Which of them are used has to be specified in the 
String dependencies. The string "X,I" for example specifies that the interface is
only space-dependent and item-dependent, i.e

    f(result, X, I)

The array dimensions specifies the expected length of result and X (if X-depdendent, otherwise will be ignored)
and quadorder determines the additional quadrature order to be used if this function is involved in some quadrature-requireing procedure.
"""
function ExtendedDataFunction(f::Function, dimensions::Array{Int,1}; name = "user data function", dependencies::String = "", quadorder::Int = 0)

    nf = (result,X,T,R,I,L) -> f(result) # no other dependencies
    if dependencies == "X"
        nf = (result,X,T,R,I,L) -> f(result, X)
    elseif dependencies == "T"
        nf = (result,X,T,R,I,L) -> f(result, T)
    elseif dependencies == "R"
        nf = (result,X,T,R,I,L) -> f(result, R)
    elseif dependencies == "I"
        nf = (result,X,T,R,I,L) -> f(result, I)
    elseif dependencies == "L"
        nf = (result,X,T,R,I,L) -> f(result, L)
    elseif dependencies == "XR"
        nf = (result,X,T,R,I,L) -> f(result, X, R)
    elseif dependencies == "XT"
        nf = (result,X,T,R,I,L) -> f(result, X, T)
    elseif dependencies == "XI"
        nf = (result,X,T,R,I,L) -> f(result, X, I)
    elseif dependencies == "XL"
        nf = (result,X,T,R,I,L) -> f(result, X, L)
    elseif dependencies == "RI"
        nf = (result,X,T,R,I,L) -> f(result, R, I)
    elseif dependencies == "XTR"
        nf = (result,X,T,R,I,L) -> f(result, X, T, R)
    elseif dependencies == "XTI"
        nf = (result,X,T,R,I,L) -> f(result, X, T, I)
    elseif dependencies == "XTL"
        nf = (result,X,T,R,I,L) -> f(result, X, T, L)
    elseif dependencies == "XRI"
        nf = (result,X,T,R,I,L) -> f(result, X, R, I)
    elseif dependencies == "XIL"
        nf = (result,X,T,R,I,L) -> f(result, X, I, L)
    elseif dependencies == "XTRIL"
        nf = (result,X,T,R,I,L) -> f(result, X, T, R, I, L)
    end

    return UserData{AbstractExtendedDataFunction,length(dimensions)}(name, dependencies, quadorder, dimensions, f, nf)
end

@inline function eval!(result, UD::UserData{AbstractActionKernel}, input, X, T, R, I, L)
    UD.negotiated_function(result, input, X, T, R, I, L)
end

@inline function eval!(result, UD::UserData{AbstractNLActionKernel}, input, input2, X, T, R, I, L)
    UD.negotiated_function(result, input, input2, X, T, R, I, L)
end

@inline function eval!(result, UD::UserData{AbstractExtendedDataFunction}, X, T, R, I, L)
    UD.negotiated_function(result, X, T, R, I, L)
end

@inline function eval!(result, UD::UserData{AbstractDataFunction}, X, T)
    UD.negotiated_function(result, X, T)
end

@inline function eval!(result, UD::UserData{AbstractDataFunction}, X, T, R, I, L)
    UD.negotiated_function(result, X, T)
end

@inline function is_xdependent(UD::UserData)
    return occursin("X", UD.dependencies)
end

@inline function is_timedependent(UD::UserData)
    return occursin("T", UD.dependencies)
end

@inline function is_itemdependent(UD::UserData)
    return occursin("I", UD.dependencies)
end

@inline function is_regiondependent(UD::UserData)
    return occursin("R", UD.dependencies)
end

@inline function is_ldependent(UD::UserData)
    return occursin("L", UD.dependencies)
end

