#= 

# 2D Commuting Interpolators
([source code](SOURCE_URL))

This example verifies a structural property of the H1 and Hdiv finite element spaces and their interpolators which is
```math
\mathrm{Curl}(I_{\mathrm{P}_k}\psi) = I_{\mathrm{RT}_{k-1}}(\mathrm{Curl}(\psi))
```
for the H1 interpolator ``I_{\mathrm{P}_k}`` and the standard Raviart-Thomas interpolator ``I_{\mathrm{RT}_{k-1}}`` for $k > 0$
which we will verify in this example for $k=1$ and $k=2$.


!!! note

    There are also higher-order commuting interpolators for $k > 2$ which will be tested as well once they are implemented.

=#

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics

## define some function
function exact_function!(result,x)
    result[1] = x[1]^2-x[2]^4 + 1
end
## and its curl = (-dy,dx) 
function exact_curl!(result,x)
    result[1] = 4*x[2]^3
    result[2] = 2*x[1]
end

## here the mesh is defined
include("../src/testgrids.jl")

## everything is wrapped in a main function
function main()

    ## choose some grid
    xgrid = uniform_refine(reference_domain(Triangle2D),1)

    ## choose commuting interpolators pair
    #FE = [H1P1{1},HDIVRT0{2}]
    FE = [H1P2{1,2},HDIVRT1{2}]

    ## do the P1 nodal interpolation of the function
    FESH1 = FESpace{FE[1]}(xgrid)
    H1Interpolation = FEVector{Float64}("H1-Interpolation",FESH1)
    interpolate!(H1Interpolation[1], exact_function!; bonus_quadorder = 4)

    ## do the RT0 interpolation of the Curl of the function
    ## since integrals over faces have to be computed exactly we need to tune the quadrature order
    FESHdiv = FESpace{FE[2]}(xgrid)
    HdivCurlInterpolation = FEVector{Float64}("Hdiv-Interpolation",FESHdiv)
    interpolate!(HdivCurlInterpolation[1], exact_curl!; bonus_quadorder = 3)

    ## evaluate the piecewise integral means of Curl of H1 interpolation
    meanCurlH1 = zeros(Float64,2,num_sources(xgrid[CellNodes]))
    meanIntegratorCurlH1 = ItemIntegrator{Float64,AssemblyTypeCELL}(CurlScalar, DoNotChangeAction(2), [0])
    evaluate!(meanCurlH1,meanIntegratorCurlH1,H1Interpolation[1])

    ## evaluate the piecewise integral means of Hdiv interpolation of Curl
    meanHdiv = zeros(Float64,2,num_sources(xgrid[CellNodes]))
    meanIntegratorHdiv = ItemIntegrator{Float64,AssemblyTypeCELL}(Identity, DoNotChangeAction(2), [0])
    evaluate!(meanHdiv,meanIntegratorHdiv,HdivCurlInterpolation[1])

    ## check their difference in the l2 vector norm
    error = sum((meanCurlH1-meanHdiv).^2, dims = 1)
    println("\n|| Curl(H1(psi) - Hdiv(Curl(psi)) ||_l2 = $error")
end

main()