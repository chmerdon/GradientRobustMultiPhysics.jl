abstract type AbstractUserDataType end

"""
````
function ∇(UD::AbstractUserDataType; quadorder = UD.quadorder - 1)
````

Provides a DataFunction with the same dependencies that evaluates the gradient of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function ∇(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
    input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[2])
    Dresult = DiffResults.JacobianResult(result_temp,input_temp)
    jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
    if is_xdependent(UD) && !is_timedependent(UD)
        cfg = ForwardDiff.JacobianConfig(UD.kernel, result_temp, input_temp)
        function data_deriv_x(result, x)
            ForwardDiff.vector_mode_jacobian!(Dresult, UD.kernel, result_temp, x, cfg)
            for j = 1 : argsizes[1], k = 1 : argsizes[2]
                result[(j-1)*argsizes[2] + k] = jac[j,k]
            end
            return nothing
        end
        return DataFunction(data_deriv_x, [argsizes[1]*argsizes[2], argsizes[2]]; name = "∇($(UD.name))", dependencies = "X", bonus_quadorder = bonus_quadorder)
    elseif is_xdependent(UD) && is_timedependent(UD)
        reduced_function_xt(t) = (result,x) -> UD.kernel(result,x,t)
        cfg = ForwardDiff.JacobianConfig(reduced_function_xt(0.0), result_temp, input_temp)
        function data_deriv_xt(result, x, t)
            ForwardDiff.vector_mode_jacobian!(Dresult, reduced_function_xt(t), result_temp, x, cfg)
            for j = 1 : argsizes[1], k = 1 : argsizes[2]
                result[(j-1)*argsizes[2] + k] = jac[j,k]
            end
            return nothing
        end
        return DataFunction(data_deriv_xt, [argsizes[1]*argsizes[2], argsizes[2]]; name = "∇($(UD.name))", dependencies = "XT", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "" || UD.dependencies == "T"
        return DataFunction(zeros(Float64, argsizes[1]*argsizes[2]); name = "∇($(UD.name))")
    else
        @error "derivatives of user functions with these dependencies not implemented yet"
    end
end

function curl3D(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    @assert argsizes[1] == 3 && argsizes[2] == 3 "curl3D needs dimensions [3,3]"
    result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
    input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[2])
    Dresult = DiffResults.JacobianResult(result_temp,input_temp)
    jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
    if is_xdependent(UD)
        cfg = ForwardDiff.JacobianConfig(UD.kernel, result_temp, input_temp)
        function data_deriv_x(result, x)
            ForwardDiff.vector_mode_jacobian!(Dresult, UD.kernel, result_temp, x, cfg)
            result[1] = jac[3,2] - jac[2,3]
            result[2] = jac[1,3] - jac[3,1]
            result[3] = jac[2,1] - jac[1,2]
            return nothing
        end
        return DataFunction(data_deriv_x, [3, 2]; name = "curl($(UD.name))", dependencies = "X", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "XT"
        reduced_function_xt(t) = (result,x) -> UD.kernel(result,x,t)
        cfg = ForwardDiff.JacobianConfig(reduced_function_xt(0.0), result_temp, input_temp)
        function data_deriv_xt(result, x, t)
            ForwardDiff.vector_mode_jacobian!(Dresult, reduced_function_xt(t), result_temp, x, cfg)
            result[1] = jac[3,2] - jac[2,3]
            result[2] = jac[1,3] - jac[3,1]
            result[3] = jac[2,1] - jac[1,2]
            return nothing
        end
        return DataFunction(data_deriv_xt, [3, 2]; name = "curl($(UD.name))", dependencies = "XT", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "" || UD.dependencies == "T"
        return DataFunction(zeros(Float64, 1); name = "curl($(UD.name))")
    else
        @error "derivatives of user functions with these dependencies not implemented yet"
    end
end

function curl2D(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    @assert argsizes[1] == 2 && argsizes[2] == 2 "curl2D needs dimensions [2,2]"
    result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
    input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[2])
    Dresult = DiffResults.JacobianResult(result_temp,input_temp)
    jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
    if is_xdependent(UD)
        cfg = ForwardDiff.JacobianConfig(UD.kernel, result_temp, input_temp)
        function data_deriv_x(result, x)
            ForwardDiff.vector_mode_jacobian!(Dresult, UD.kernel, result_temp, x, cfg)
            result[1] = jac[1,2] - jac[2,1]
            return nothing
        end
        return DataFunction(data_deriv_x, [1, 2]; name = "curl($(UD.name))", dependencies = "X", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "XT"
        reduced_function_xt(t) = (result,x) -> UD.kernel(result,x,t)
        cfg = ForwardDiff.JacobianConfig(reduced_function_xt(0.0), result_temp, input_temp)
        function data_deriv_xt(result, x, t)
            ForwardDiff.vector_mode_jacobian!(Dresult, reduced_function_xt(t), result_temp, x, cfg)
            result[1] = jac[1,2] - jac[2,1]
            return nothing
        end
        return DataFunction(data_deriv_xt, [1,2]; name = "curl($(UD.name))", dependencies = "XT", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "" || UD.dependencies == "T"
        return DataFunction(zeros(Float64, 1); name = "curl($(UD.name))")
    else
        @error "derivatives of user functions with these dependencies not implemented yet"
    end
end


function curl_scalar(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    @assert argsizes[1] == 1 && argsizes[2] == 2 "curl_scalar needs dimensions [1,2]"
    result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
    input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[2])
    Dresult = DiffResults.JacobianResult(result_temp,input_temp)
    jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
    if is_xdependent(UD)
        cfg = ForwardDiff.JacobianConfig(UD.kernel, result_temp, input_temp)
        function data_deriv_x(result, x)
            ForwardDiff.vector_mode_jacobian!(Dresult, UD.kernel, result_temp, x, cfg)
            result[1] = -jac[1,2]
            result[2] = jac[1,1]
            return nothing
        end
        return DataFunction(data_deriv_x, [2, 2]; name = "curl($(UD.name))", dependencies = "X", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "XT"
        reduced_function_xt(t) = (result,x) -> UD.kernel(result,x,t)
        cfg = ForwardDiff.JacobianConfig(reduced_function_xt(0.0), result_temp, input_temp)
        function data_deriv_xt(result, x, t)
            ForwardDiff.vector_mode_jacobian!(Dresult, reduced_function_xt(t), result_temp, x, cfg)
            result[1] = -jac[1,2]
            result[2] = jac[1,1]
            return nothing
        end
        return DataFunction(data_deriv_xt, [2,2]; name = "curl($(UD.name))", dependencies = "XT", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "" || UD.dependencies == "T"
        return DataFunction(zeros(Float64, 2); name = "curl($(UD.name))")
    else
        @error "derivatives of user functions with these dependencies not implemented yet"
    end
end

"""
````
function curl(UD::AbstractUserDataType; quadorder = UD.quadorder - 1)
````

Provides a DataFunction with the same dependencies that evaluates the curl of the DataFunction UD. The derivatives are computed by ForwardDiff.
Depending on the dimensions of UD, either CurlScalar (UD.argsizes == [1,2]), Curl2D (UD.argsizes == [2,2]) or Curl3D (UD.argsizes == [3,3])
is generated.
"""
function curl(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    if argsizes[1] == 1 # 2D curl of scalar-function curl(f: R^2 --> R^1) : R^2 --> R^2 = [-df/dy, df/dx]
        curl_scalar(UD; bonus_quadorder = bonus_quadorder)
    elseif argsizes[1] == 2 # 2D curl of vector-valued function curl(f: R^2 --> R^2) : R^2 --> R^1 = -df1/dy + df2/dx
        curl2D(UD; bonus_quadorder = bonus_quadorder)
    elseif argsizes[1] == 3 # 3D curl of vector-valued function curl(f: R^3 --> R^3) : R^2 --> R^1 = ∇ × f
        curl3D(UD; bonus_quadorder = bonus_quadorder)
    else
        @error "curl for these function dimensions not known/implemented"
    end
end

"""
````
function div(UD::AbstractUserDataType; quadorder = UD.quadorder - 1)
````

Provides a DataFunction with the same dependencies that evaluates the divergence of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function Base.div(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    @assert argsizes[1] == argsizes[2] "div needs equal dimensions"
    result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
    input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[2])
    Dresult = DiffResults.JacobianResult(result_temp,input_temp)
    jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
    if is_xdependent(UD)
        cfg = ForwardDiff.JacobianConfig(UD.kernel, result_temp, input_temp)
        function data_deriv_x(result, x)
            ForwardDiff.vector_mode_jacobian!(Dresult, UD.kernel, result_temp, x, cfg)
            result[1] = 0
            for j = 1 : argsizes[2]
                result[1] += jac[j,j]
            end
            return nothing
        end
        return DataFunction(data_deriv_x, [1, argsizes[2]]; name = "div($(UD.name))", dependencies = "X", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "XT"
        reduced_function_xt(t) = (result,x) -> UD.kernel(result,x,t)
        cfg = ForwardDiff.JacobianConfig(reduced_function_xt(0.0), result_temp, input_temp)
        function data_deriv_xt(result, x, t)
            ForwardDiff.vector_mode_jacobian!(Dresult, reduced_function_xt(t), result_temp, x, cfg)
            result[1] = 0
            for j = 1 : argsizes[2]
                result[1] += jac[j,j]
            end
            return nothing
        end
        return DataFunction(data_deriv_xt, [1, argsizes[2]]; name = "div($(UD.name))", dependencies = "XT", bonus_quadorder = bonus_quadorder)
    elseif UD.dependencies == "" || UD.dependencies == "T"
        return DataFunction(zeros(Float64, 1); name = "div($(UD.name))")
    else
        @error "derivatives of user functions with these dependencies not implemented yet"
    end
end


"""
````
function nodevalues(nodevals, xgrid::ExtendableGrid{Tv,Ti}, UD::UserData; time = 0) where {Tv,Ti}
````

Returns a 2D array with the node values of the data function for the given grid.
"""
function nodevalues(xgrid::ExtendableGrid{Tv,Ti}, UD::AbstractUserDataType; T = Float64, time = 0) where {Tv,Ti}
    xCoordinates::Array{Tv,2} = xgrid[Coordinates]
    nnodes::Int = size(xCoordinates,2)

    set_time!(UD, time)
    if is_xdependent(UD) && length(UD.x) != size(xCoordinates,1)
        UD.x = zeros(T,size(xCoordinates,1))
    end
    nodevals = zeros(T,UD.argsizes[1],nnodes)
    for j = 1 : nnodes
        if is_itemdependent(UD)
            UD.citem[1] = cell
            UD.citem[2] = cell
            UD.citem[3] = xCellRegions[cell]
        end
        if is_xdependent(UD)
            UD.x .= view(xCoordinates,:,j)
        end
        eval_data!(UD)
        for k = 1 : UD.argsizes[1]
            nodevals[k,j] = UD.val[k]
        end
    end
    return nodevals
end




mutable struct DataFunction{T,Ti,dx,dt,di,dl,ndim,KernelType} <: AbstractUserDataType
    name::String                    # some name used in info messages etc.
    kernel::KernelType              # kernel function
    argsizes::SVector{ndim,Int}     # argument sizes
    x::Vector{T}                    # current space coordinates
    xref::Vector{T}                 # current local coordinate
    item::Array{Ti,1}               # contains item number, parent item, region number
    time::T                         # current time
    bonus_quadorder::Int            # causes increase of quadorder in any used integration
    val::Array{T,1}                 # result vector
end

set_time!(A::DataFunction, time) = (A.time = time)

is_xdependent(::DataFunction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = dx
is_timedependent(::DataFunction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = dt
is_itemdependent(::DataFunction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = di
is_xrefdependent(::DataFunction{T,Ti,dx,dt,di,dl,ndim}) where {T,Ti,dx,dt,di,dl,ndim} = dl
function dependencies(A::DataFunction; enforce = "")
    dependencies = occursin("X", enforce) ? "X" : is_xdependent(A) ? "X" : ""
    dependencies *= occursin("T", enforce) ? "T" : is_timedependent(A) ? "T" : ""
    dependencies *= occursin("I", enforce) ? "I" : is_itemdependent(A) ? "I" : ""
    dependencies *= occursin("L", enforce) ? "L" : is_xrefdependent(A) ? "L" : ""
    return dependencies
end
function couple!(A::DataFunction,B::DataFunction)
    A.xref = B.xref
    A.x = B.x
    A.item = B.item
end
    


"""
$(TYPEDSIGNATURES)

generates a DataFunction that can be used in the construction of PDEoperators, interpolations etc. and essentially consists of a kernel function
specified by the user plus additional information on argument dimensions and additional dependencies:

- kernel   : Function with interface (result, ...)
- argsizes : expected lengths of [result, interface]

Optional arguments:

- dependencies    : substring of "XTIL" that specifies if the kernel also depends on space coordinates (X), time (T), item (I), local coordinates (L)
- bonus_quadorder : is added to the quadrature order computed based on the used FESpaces during assembly
- name            : name of this Action used in print messages
- Tv              : expected NumberType for result/input
- Ti              : expected NumberType for grid enumeration infos (e.g. item/region numbers when "I" dependecy is used)
"""
function DataFunction(kernel::Function, argsizes; Tv = Float64, Ti = Int32, dependencies = "", bonus_quadorder = 0, name = "user action")
    dx = occursin("X", dependencies)
    dt = occursin("T", dependencies)
    di = occursin("I", dependencies)
    dl = occursin("L", dependencies)
    return DataFunction{Tv,Ti,dx,dt,di,dl,length(argsizes),typeof(kernel)}(
        name, kernel, argsizes, zeros(Tv, 3), zeros(Tv, 3), zeros(Ti,5), 0, bonus_quadorder, zeros(Tv, argsizes[1]))
end

"""
````
function DataFunction(c::Array{<:Real,1}; name = "constant user data", quadorder::Int = 0)
````

Directly generates a DataFunction from a given array c, i.e. a DataFunction that is constant and has no dependencies on x or t.
"""
function DataFunction(c::Array{<:Real,1}; name = "auto", bonus_quadorder::Int = 0)
    dimensions = [length(c),0]
    function f_from_c(result)
        result .= c
    end
    if name == "auto"
        name = "Constant($c)"
    end

    return DataFunction(f_from_c, dimensions; name = name, dependencies = "", bonus_quadorder = bonus_quadorder)
end

function eval_data!(A::DataFunction{T,Ti,false,false,false,false}) where {T,Ti}
    A.kernel(A.val)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,false,false,false}) where {T,Ti}
    A.kernel(A.val, A.x)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,true,false,false}) where {T,Ti}
    A.kernel(A.val, A.x, A.time)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,true,false,false}, x, t) where {T,Ti}
    A.kernel(A.val, x, t)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,false,false,false}, x) where {T,Ti}
    A.kernel(A.val, x)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,false,true,false,false}) where {T,Ti}
    A.kernel(A.val, A.time)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,false,false,false,true}) where {T,Ti}
    A.kernel(A.val, A.xref)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,false,false,true}) where {T,Ti}
    A.kernel(A.val, A.x, A.xref)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,false,true,false}) where {T,Ti}
    A.kernel(A.val, A.x, A.item)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,true,true,false}) where {T,Ti}
    A.kernel(A.val, A.x, A.time, A.item)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,true,false,true}) where {T,Ti}
    A.kernel(A.val, A.x, A.time, A.xref)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,false,true,true}) where {T,Ti}
    A.kernel(A.val, A.x, A.item, A.xref)
    return nothing
end
function eval_data!(A::DataFunction{T,Ti,true,true,true,true}) where {T,Ti}
    A.kernel(A.val, A.x, A.time, A.item, A.xref)
    return nothing
end

###########################################
## STUFF USED BY NONLINEAR FORM ASSEMBLY ##
###########################################


abstract type AbstractNonlinearFormHandler end

mutable struct OperatorWithUserJacobian{Tv,Ti,dx,dt,di,ndim,OType,JType} <: AbstractNonlinearFormHandler
    name::String
    operator::OType
    jacobian::JType
    argsizes::SVector{ndim,Int}
    x::Vector{Tv}
    item::Array{Ti,1} # item, parent, region
    time::Tv
    bonus_quadorder::Int
    jac::Array{Tv,2}
    val::Array{Tv,1}
end

function OperatorWithUserJacobian(o, j, argsizes; name = "", Ti = Int32, dependencies = "", bonus_quadorder = 0, sparse_jacobian::Bool = false)
    dx = dependencies in ["X","XT","XI","XTI"]
    dt = dependencies in ["T","XT","TI","XTI"]
    di = dependencies in ["I","XI","TI","XTI"]
    if sparse_jacobian
        result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
        input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[3])

        if dependencies == "X" # x-dependent
            o_x(x) = (result,input) -> o(result,input,x)
            config_eval = o_x([1.0,1.0,1.0])
        elseif dependencies == "T" # time-dependent
            o_t(t) = (result,input) -> o(result,input,t)
            config_eval = o_t(0.0)
        elseif dependencies == "I" # region-dependent
            o_r(r) = (result,input) -> o(result,input,r)
            config_eval = o_r(0)
        elseif dependencies == "XT" # xt-dependent
            o_xt(x,t) = (result,input) -> o(result,input,x,t)
            config_eval = o_xt([1.0,1.0,1.0],0.0)
        elseif dependencies == "XI" # xr-dependent
            o_xr(x,r) = (result,input) -> o(result,input,x,r)
            config_eval = o_xr([1.0,1.0,1.0],0)
        elseif dependencies == "TI" # tr-dependent
            o_tr(t,r) = (result,input) -> o(result,input,t,r)
            config_eval = o_tr(0.0,0)
        elseif dependencies == "XTI" # xtr-dependent
            o_xtr(x,t,r) = (result,input) -> o(result,input,x,t,r)
            config_eval = o_xtr([1.0,1.0,1.0],0.0,0)
        else
            config_eval = o
        end
        sparsity_pattern = jacobian_sparsity(config_eval,result_temp,input_temp)
        jac = Float64.(sparse(sparsity_pattern))
    else
        jac = zeros(Float64,argsizes[1],argsizes[2])
    end
    return OperatorWithUserJacobian{Float64,Ti,dx,dt,di,length(argsizes),typeof(o),typeof(j)}(name,o,j,argsizes,zeros(Float64,3),zeros(Ti,5),0,bonus_quadorder,jac,zeros(Float64,argsizes[1]))
end

set_time!(J::OperatorWithUserJacobian, time) = (J.time = time)
is_xdependent(J::OperatorWithUserJacobian{T,Ti,dx,dt,dr,ndim}) where {T,Ti,dx,dt,dr,ndim} = dx
is_timedependent(J::OperatorWithUserJacobian{T,Ti,dx,dt,dr,ndim}) where {T,Ti,dx,dt,dr,ndim} = dt
is_itemdependent(J::OperatorWithUserJacobian{T,Ti,dx,dt,dr,ndim}) where {T,Ti,dx,dt,dr,ndim} = dr

function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,false,false,false}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current)
    J.operator(J.val, input_current)
    return nothing
end
function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,true,false,false}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current, J.x)
    J.operator(J.val, input_current, J.x)
    return nothing
end
function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,true,false,true}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current, J.x, C.item)
    J.operator(J.val, input_current, J.x, C.item)
    return nothing
end
function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,false,true,false}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current, C.time)
    J.operator(J.val, input_current, C.time)
    return nothing
end
function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,false,true,true}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current, C.time, C.item)
    J.operator(J.val, input_current, C.time, C.item)
    return nothing
end
function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,false,false,true}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current, C.item)
    J.operator(J.val, input_current, C.item)
    return nothing
end
function eval_jacobian!(J::OperatorWithUserJacobian{T,Ti,true,true,true}, input_current) where {T,Ti}
    J.jacobian(J.jac, input_current, J.x, C.time, C.item)
    J.operator(J.val, input_current, J.x, C.time, C.item)
    return nothing
end


mutable struct OperatorWithADJacobian{T,Ti,dx,dt,di,sparse,ndim,OType,JType,JacType} <: AbstractNonlinearFormHandler
    name::String
    operator::OType
    jacobian::JType
    argsizes::SVector{ndim,Int}
    x::Vector{T}
    item::Array{Ti,1} # item, parent, region
    time::T
    bonus_quadorder::Int # modifies the number generated from operators and FEspaces of NonlinearForm
    Dresult::DiffResults.DiffResult
    cfg::Union{ForwardDiff.JacobianConfig,SparseDiffTools.ForwardColorJacCache}
    jac::JacType
    val::Array{T,1}
end

function OperatorWithADJacobian(o, argsizes; name = "", Ti = Int32, dependencies = "", bonus_quadorder = 0, sparse_jacobian::Bool = false)
    result_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[1])
    input_temp::Array{Float64,1} = Vector{Float64}(undef,argsizes[3])

    dx = dependencies in ["X","XT","XI","XTI"]
    dt = dependencies in ["T","XT","TI","XTI"]
    di = dependencies in ["I","XI","TI","XTI"]

    if dependencies == "X" # x-dependent
        o_x(x) = (result,input) -> o(result,input,x)
        config_eval = o_x([1.0,1.0,1.0])
        negotiated_o = o_x
    elseif dependencies == "T" # time-dependent
        o_t(t) = (result,input) -> o(result,input,t)
        config_eval = o_t(0.0)
        negotiated_o = o_t
    elseif dependencies == "I" # region-dependent
        o_r(r) = (result,input) -> o(result,input,r)
        config_eval = o_r(zeros(Ti,5))
        negotiated_o = o_r
    elseif dependencies == "XT" # xt-dependent
        o_xt(x,t) = (result,input) -> o(result,input,x,t)
        config_eval = o_xt([1.0,1.0,1.0],0.0)
        negotiated_o = o_xt
    elseif dependencies == "XI" # xr-dependent
        o_xr(x,r) = (result,input) -> o(result,input,x,r)
        config_eval = o_xr([1.0,1.0,1.0],zeros(Ti,5))
        negotiated_o = o_xr
    elseif dependencies == "TI" # tr-dependent
        o_tr(t,r) = (result,input) -> o(result,input,t,r)
        config_eval = o_tr(0.0,zeros(Ti,5))
        negotiated_o = o_tr
    elseif dependencies == "XTI" # xtr-dependent
        o_xtr(x,t,r) = (result,input) -> o(result,input,x,t,r)
        config_eval = o_xtr([1.0,1.0,1.0],0.0,zeros(Ti,5))
        negotiated_o = o_xtr
    else
        negotiated_o = o
        config_eval = o
    end

    Dresult = DiffResults.JacobianResult(result_temp,input_temp)
    temp::Array{Float64,1} = DiffResults.value(Dresult)
    if sparse_jacobian
        sparsity_pattern = jacobian_sparsity(config_eval,result_temp,input_temp)
        jac = Float64.(sparse(sparsity_pattern))
        colors = matrix_colors(jac)
        cfg = ForwardColorJacCache(config_eval,input_temp,nothing;
                    dx = nothing,
                    colorvec = colors,
                    sparsity = nothing)
    else
        jac::Array{Float64,2} = DiffResults.jacobian(Dresult)
        colors = nothing
        cfg = ForwardDiff.JacobianConfig(config_eval, result_temp, input_temp, ForwardDiff.Chunk{argsizes[3]}())
    end

    return OperatorWithADJacobian{Float64,Ti,dx,dt,di,sparse_jacobian,length(argsizes),typeof(negotiated_o),typeof(NothingFunction),typeof(jac)}(name,negotiated_o,NothingFunction,argsizes,zeros(Float64,3),zeros(Ti,5),0.0,bonus_quadorder,Dresult,cfg,jac,temp)
end

set_time!(J::OperatorWithADJacobian, time) = (J.time = time)
is_xdependent(J::OperatorWithADJacobian{T,Ti,dx,dt,dr,ndim}) where {T,Ti,dx,dt,dr,ndim} = dx
is_timedependent(J::OperatorWithADJacobian{T,Ti,dx,dt,dr,ndim}) where {T,Ti,dx,dt,dr,ndim} = dt
is_itemdependent(J::OperatorWithADJacobian{T,Ti,dx,dt,dr,ndim}) where {T,Ti,dx,dt,dr,ndim} = dr

function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,false,false,false,false}, input_current) where {T,Ti}
    J.Dresult = ForwardDiff.chunk_mode_jacobian!(J.Dresult, J.operator, J.val, input_current, J.cfg)
    return nothing
end
function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,false,false,false,true}, input_current) where {T,Ti}
    forwarddiff_color_jacobian!(J.jac, J.operator, input_current, J.cfg)
    J.operator(J.val, input_current)
    return nothing
end

function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,true,false,false,false}, input_current) where {T,Ti}
    J.Dresult = ForwardDiff.chunk_mode_jacobian!(J.Dresult, J.operator(J.x), J.val, input_current, J.cfg)
    return nothing
end
function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,true,false,false,true}, input_current) where {T,Ti}
    forwarddiff_color_jacobian!(J.jac, J.operator(J.x), input_current, J.cfg)
    J.operator(J.x)(J.val, input_current)
    return nothing
end

function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,false,true,false,false}, input_current) where {T,Ti}
    J.Dresult = ForwardDiff.chunk_mode_jacobian!(J.Dresult, J.operator(J.time), J.val, input_current, J.cfg)
    return nothing
end
function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,false,true,false,true}, input_current) where {T,Ti}
    forwarddiff_color_jacobian!(J.jac, J.operator(J.time), input_current, J.cfg)
    J.operator(J.time)(J.val, input_current)
    return nothing
end

function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,false,false,true,false}, input_current) where {T,Ti}
    J.Dresult = ForwardDiff.chunk_mode_jacobian!(J.Dresult, J.operator(J.item), J.val, input_current, J.cfg)
    return nothing
end
function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,false,false,true,true}, input_current) where {T,Ti}
    forwarddiff_color_jacobian!(J.jac, J.operator(J.item), input_current, J.cfg)
    J.operator(J.item)(J.val, input_current)
    return nothing
end

function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,true,true,true,false}, input_current) where {T,Ti}
    J.Dresult = ForwardDiff.chunk_mode_jacobian!(J.Dresult, J.operator(J.x, J.item), J.val, input_current, J.cfg)
    return nothing
end
function eval_jacobian!(J::OperatorWithADJacobian{T,Ti,true,true,true,true}, input_current) where {T,Ti}
    forwarddiff_color_jacobian!(J.jac, J.operator(J.x, J.time, J.item), input_current, J.cfg)
    J.operator(x, J.time, J.item)(J.val, input_current)
    return nothing
end