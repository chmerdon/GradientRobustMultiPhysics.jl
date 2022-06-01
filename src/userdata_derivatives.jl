
###############################################
## FORWARDDIFF WRAPPERS FOR AUTO-DERIVATIVES ##
###############################################

function config_derivatives!(UD::DataFunction, x_order = 1, t_order = 0)
    argsizes = UD.argsizes
    @assert x_order > 0 || t_order > 0
    result = zeros(Real, argsizes[1]) # result for u

    if x_order > 0
        @assert x_order in 1:2
        result1 = zeros(Float64, argsizes[1], argsizes[2]) # result for grad(u)
        if !is_xdependent(UD)
            result2 = zeros(Float64, argsizes[1]*argsizes[2], argsizes[2])
            UD.eval_derivs = [(x) -> result1, (x) -> result2]
        else
            Dresult1 = DiffResults.DiffResult(result,result1)
            mask_x = 1:argsizes[2]
            if is_xdependent(UD) && !is_timedependent(UD)
                data_wrap_x = x -> (
                    fill!(result,0);
                    UD.kernel(result, x);
                    return result;
                )
            else
                data_wrap_x = x -> (
                    fill!(result,0);
                    UD.kernel(result, x, UD.time);
                    return result;
                )
            end
            Dcfg = ForwardDiff.JacobianConfig(data_wrap_x, result1)

            eval_J = x -> (
                ForwardDiff.jacobian!(Dresult1, data_wrap_x, view(x, mask_x));
                return DiffResults.jacobian(Dresult1);
                )

            if x_order == 1
                UD.eval_derivs = [eval_J]
            elseif x_order == 2
                result12 = zeros(Real, argsizes[1], argsizes[2]) # result for grad(u)
                Dresult12 = DiffResults.DiffResult(result,result12)
                eval_J2 = x -> (
                    ForwardDiff.jacobian!(Dresult12, data_wrap_x, view(x, mask_x));
                    return DiffResults.jacobian(Dresult12);
                    )

                result2 = zeros(Float64, argsizes[1]*argsizes[2], argsizes[2]) # result for H(u)
                Dresult2 = DiffResults.DiffResult(result,result2)
            # Dcfg2 = ForwardDiff.JacobianConfig(eval_J2, result2)

                eval_H = x -> (
                    ForwardDiff.jacobian!(Dresult2, eval_J2, view(x, mask_x));
                    return DiffResults.jacobian(Dresult2);
                )
                UD.eval_derivs = [eval_J, eval_H]
            end
        end
    end
    if t_order > 0
        @assert t_order == 1
        result_t1 = zeros(Float64, argsizes[1]) # result for dt(u)
        @assert is_timedependent(UD)
        if is_xdependent(UD) && is_timedependent(UD)
            data_wrap_t = t -> (
                fill!(result,0);
                UD.kernel(result, UD.x, t);
                return result;
            )
        else
            data_wrap_t = t -> (
                fill!(result,0);
                UD.kernel(result, t);
                return result;
            )
        end
        Dcfg = ForwardDiff.GradientConfig(data_wrap_t, result_t1)

        evalt_Jt = t -> (
            ForwardDiff.derivative!(result_t1, data_wrap_t, t);
            return result_t1;
            )

        UD.eval_derivs_t = [evalt_Jt]
    end
    return nothing
end

function eval_derivatives!(UD::DataFunction, order, x)
    UD.eval_derivs[order](x)
end
function eval_derivatives!(UD::DataFunction, order, x, t)
    UD.time = t
    UD.eval_derivs[order](x)
end

function eval_derivatives_t!(UD::DataFunction, order, t)
    UD.eval_derivs_t[order](t)
end
function eval_derivatives_t!(UD::DataFunction, order, x, t)
    UD.x = x
    UD.eval_derivs_t[order](t)
end

"""
````
function eval_∇(UD::AbstractUserDataType) -> Function
````

Provides a function that evaluates and returns the gradient of the DataFunction UD in x (and t if UD
depends on time). The derivatives are computed by ForwardDiff.
"""
function eval_∇(UD::DataFunction)
    if UD.eval_derivs === nothing || length(UD.eval_derivs) < 1
        config_derivatives!(UD, 1)
    end
    function closure(args...)::Matrix{Float64}
        return eval_derivatives!(UD::DataFunction, 1, args...);
    end
end


"""
````
function eval_dt(UD::AbstractUserDataType) -> Function
````

Provides a function that evaluates and returns the gradient of the DataFunction UD in x (and t if UD
depends on time). The derivatives are computed by ForwardDiff.
"""
function eval_dt(UD::DataFunction)
    if UD.eval_derivs_t === nothing || length(UD.eval_derivs_t) < 1
        config_derivatives!(UD, 0, 1)
    end
    function closure(args...)::Vector{Float64}
        return eval_derivatives_t!(UD::DataFunction, 1, args...);
    end
end

"""
````
function dt(UD::AbstractUserDataType; quadorder = UD.quadorder - 1) -> DataFunction
````

Provides a DataFunction with the same dependencies that evaluates the gradient of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function dt(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    if is_timedependent(UD)
        dtUD = eval_dt(UD)
        return DataFunction((result, args...) -> (result .= view(dtUD(args...),:);), [argsizes[1], argsizes[2]]; name = "dt($(UD.name))", dependencies = dependencies(UD), bonus_quadorder = bonus_quadorder)
    else
        return DataFunction(zeros(Float64, argsizes[1]); name = "dt($(UD.name))")
    end
end

"""
````
function ∇(UD::AbstractUserDataType; quadorder = UD.quadorder - 1) -> DataFunction
````

Provides a DataFunction with the same dependencies that evaluates the gradient of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function ∇(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    argsizes = UD.argsizes
    if is_xdependent(UD)
        ∇UD = eval_∇(UD)
        return DataFunction((result, args...) -> (result .= view(transpose(∇UD(args...)),:);), [argsizes[1]*argsizes[2], argsizes[2]]; name = "∇($(UD.name))", dependencies = dependencies(UD), bonus_quadorder = bonus_quadorder)
    else
        return DataFunction(zeros(Float64, argsizes[1]*argsizes[2]); name = "∇($(UD.name))")
    end
end


"""
````
function eval_∇(UD::AbstractUserDataType) -> Function
````

Provides a function that evaluates and returns the Hessian of the DataFunction UD in x (and t if UD
depends on time). The derivatives are computed by ForwardDiff.
"""
function eval_H(UD::DataFunction)
    if UD.eval_derivs === nothing || length(UD.eval_derivs) < 2
        config_derivatives!(UD, 2)
    end
    argsizes = UD.argsizes
    function closure(args...)::Matrix{Float64}
        return eval_derivatives!(UD::DataFunction, 2, args...);
    end
end


"""
````
function H(UD::AbstractUserDataType; quadorder = UD.quadorder - 2) -> DataFunction
````

Provides a DataFunction with the same dependencies that evaluates the Hessian of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function H(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 2)
    HUD = eval_H(UD)
    argsizes = UD.argsizes

    function Hreshape!(result, H)
        for u = 1 : argsizes[1]
            for xj = 1 : argsizes[2]
                for xk = 1 : argsizes[2]
                    result[(u-1)*argsizes[2]*argsizes[2] + (xj-1)*argsizes[2] + xk] = H[(xj-1)*argsizes[1] + u, xk]
                end
            end
        end
        return nothing
    end

    return DataFunction((result, args...) -> (Hreshape!(result, HUD(args...));), [argsizes[1]*argsizes[2]*argsizes[2], argsizes[2]]; name = "H($(UD.name))", dependencies = dependencies(UD), bonus_quadorder = bonus_quadorder)
end

"""
````
function eval_div(UD::AbstractUserDataType) -> Function
````

Provides a function that evaluates and returns the divergence of the DataFunction UD in x (and t if UD
depends on time). The derivatives are computed by ForwardDiff.
"""
function eval_div(UD::DataFunction)
    if UD.eval_derivs === nothing || length(UD.eval_derivs) < 1
        config_derivatives!(UD, 1)
    end
    argsizes = UD.argsizes
    result = zeros(Float64, 1)
    function closure(args...)::Vector{Float64}
        jac::Matrix{Float64} = eval_derivatives!(UD::DataFunction, 1, args...);
        result[1] = 0
        for j = 1 : size(jac,1)
            result[1] += jac[j,j]
        end
        return result
    end
end

"""
````
function Base.div(UD::AbstractUserDataType; quadorder = UD.quadorder - 1)
````

Provides a DataFunction with the same dependencies that evaluates the divergence of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function Base.div(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    divUD = eval_div(UD)
    argsizes = UD.argsizes
    return DataFunction((result, args...) -> (result .= view(divUD(args...),:);), [1, argsizes[2]]; name = "div($(UD.name))", dependencies = dependencies(UD), bonus_quadorder = bonus_quadorder)
end

"""
````
function eval_∇(UD::AbstractUserDataType) -> Function
````

Provides a function that evaluates and returns the Laplacian of the DataFunction UD in x (and t if UD
depends on time). The derivatives are computed by ForwardDiff.
"""
function eval_Δ(UD::DataFunction)
    if UD.eval_derivs === nothing || length(UD.eval_derivs) < 2
        config_derivatives!(UD, 2)
    end
    argsizes = UD.argsizes
    result = zeros(Float64, argsizes[1])
    function closure(args...)::Vector{Float64}
        H::Matrix{Float64} = eval_derivatives!(UD::DataFunction, 2, args...);
        fill!(result,0);
        for j = 1 : argsizes[1], k = 1 : argsizes[2]
            result[j] += H[(k-1)*argsizes[1] + j, k];
        end
        return result
    end
end

"""
````
function Δ(UD::AbstractUserDataType; quadorder = UD.quadorder - 2)
````

Provides a DataFunction with the same dependencies that evaluates the Laplacian of the DataFunction UD.
The derivatives are computed by ForwardDiff.
"""
function Δ(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 2)
    ΔUD = eval_Δ(UD)
    argsizes = UD.argsizes
    return DataFunction((result, args...) -> (result .= view(ΔUD(args...),:);), [argsizes[1], argsizes[2]]; name = "Δ($(UD.name))", dependencies = dependencies(UD), bonus_quadorder = bonus_quadorder)
end


"""
````
function eval_curl(UD::AbstractUserDataType) -> Function
````

Provides a function that evaluates and returns the curl of the DataFunction UD in x (and t if UD
depends on time). Which kind of curl is returned depends on argsizes[1]. For argsizes[1] = 1,
the curl for scalar-valued functions in 2D is calculated, if argsizes[1] = 2 the curl for
vector fields in 2D is calculated, and for argsies[1] = 3 the 3D curl is calculated.
The derivatives are computed by ForwardDiff.
"""
function eval_curl(UD::DataFunction)
    if UD.eval_derivs === nothing || length(UD.eval_derivs) < 1
        config_derivatives!(UD, 1)
    end
    argsizes = UD.argsizes

    if argsizes[1] == 1 # 2D curl of scalar-function curl(f: R^2 --> R^1) : R^2 --> R^2 = [-df/dy, df/dx]
        result = zeros(Float64, 2)
        function closure_curl1D(args...)::Vector{Float64}
            jac::Matrix{Float64} = eval_derivatives!(UD::DataFunction, 1, args...)
            result[1] = -jac[1,2]
            result[2] = jac[1,1]
            return result
        end
        return closure_curl1D
    elseif argsizes[1] == 2 # 2D curl of vector-valued function curl(f: R^2 --> R^2) : R^2 --> R^1 = -df1/dy + df2/dx
        result = zeros(Float64, 1)
        function closure_curl2D(args...)::Vector{Float64}
            jac::Matrix{Float64} = eval_derivatives!(UD::DataFunction, 1, args...)
            result[1] = jac[1,2] - jac[2,1]
            return result
        end
    elseif argsizes[1] == 3 # 3D curl of vector-valued function curl(f: R^3 --> R^3) : R^2 --> R^1 = ∇ × f
        result = zeros(Float64, 3)
        function closure_curl3D(args...)::Vector{Float64}
            jac::Matrix{Float64} = eval_derivatives!(UD::DataFunction, 1, args...)
            result[1] = jac[3,2] - jac[2,3]
            result[2] = jac[1,3] - jac[3,1]
            result[3] = jac[2,1] - jac[1,2]
            return result
        end
    else
        @error "curl for these function dimensions not known/implemented"
    end
end

"""
````
function curl(UD::AbstractUserDataType; quadorder = UD.quadorder - 1)
````

Provides a DataFunction with the same dependencies that evaluates the curl of the DataFunction UD. The derivatives are computed by ForwardDiff.
"""
function curl(UD::AbstractUserDataType; bonus_quadorder = UD.bonus_quadorder - 1)
    curlUD = eval_curl(UD)
    argsizes = UD.argsizes
    if argsizes[1] == 1 # 2D curl of scalar-function curl(f: R^2 --> R^1) : R^2 --> R^2 = [-df/dy, df/dx]
        resultdim = 2
    elseif argsizes[1] == 2 # 2D curl of vector-valued function curl(f: R^2 --> R^2) : R^2 --> R^1 = -df1/dy + df2/dx
        resultdim = 1
    elseif argsizes[1] == 3 # 3D curl of vector-valued function curl(f: R^3 --> R^3) : R^2 --> R^1 = ∇ × f
        resultdim = 3
    else
        @error "curl for these function dimensions not known/implemented"
    end
    return DataFunction((result, args...) -> (result .= view(curlUD(args...),:);), [resultdim, argsizes[2]]; name = "div($(UD.name))", dependencies = dependencies(UD), bonus_quadorder = bonus_quadorder)
end
    
