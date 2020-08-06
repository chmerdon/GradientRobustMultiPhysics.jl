#= 

# 2D Commuting Interpolators
([source code](SOURCE_URL))

This example verifies a structural property of the lowest-order H1 and Hdiv finite element spaces and their interpolators which is
```math
\mathrm{Curl}(I_{\mathrm{P1}}\psi) = I_{\mathrm{RT0}}(\mathrm{Curl}(\psi))
```
for the nodal interpolator ``I_{\mathrm{P1}}`` into the pieceise linear polynomials and the lowest-order standard Raviart-Thomas interpolator ``I_{\mathrm{RT0}}``
which we will verify in this example by comparing the nodal values of both sides of the equation. Note, that both expression are piecewise
constant, so we compare nodalvalues of some discontinuous quantity. Nevertheless, since both quantities are (hopefully) the same discontinuous function,
their nodal values should be identical.


!!! note

    There are also higher-order commuting interpolators which will be tested as well once they are implemented.

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

    ## do the P1 nodal interpolation of the function
    FESH1 = FESpace{H1P1{1}}(xgrid)
    H1Interpolation = FEVector{Float64}("H1-Interpolation",FESH1)
    interpolate!(H1Interpolation[1], exact_function!)

    ## do the RT0 interpolation of the Curl of the function
    ## since integrals over faces have to be computed exactly we need to tune the quadrature order
    FESHdiv = FESpace{HDIVRT0{2}}(xgrid)
    HdivCurlInterpolation = FEVector{Float64}("Hdiv-Interpolation",FESHdiv)
    interpolate!(HdivCurlInterpolation[1], exact_curl!; bonus_quadorder = 3)

    ## evaluate the piecewise integral means of Curl of H1 interpolation
    meanCurlH1 = zeros(Float64,num_sources(xgrid[CellNodes]),2)
    meanIntegratorCurlH1 = ItemIntegrator{Float64,AssemblyTypeCELL}(CurlScalar, DoNotChangeAction(2), [0])
    evaluate!(meanCurlH1,meanIntegratorCurlH1,H1Interpolation[1])

    ## evaluate the piecewise integral means of Hdiv interpolation of Curl
    meanHdiv = zeros(Float64,num_sources(xgrid[CellNodes]),2)
    meanIntegratorHdiv = ItemIntegrator{Float64,AssemblyTypeCELL}(Identity, DoNotChangeAction(2), [0])
    evaluate!(meanHdiv,meanIntegratorHdiv,HdivCurlInterpolation[1])

    ## check their difference in the l2 vector norm
    error = sum((meanCurlH1-meanHdiv).^2, dims = 1)
    println("\n|| Curl(H1(psi) - Hdiv(Curl(psi)) ||_l2 = $error")
end

main()