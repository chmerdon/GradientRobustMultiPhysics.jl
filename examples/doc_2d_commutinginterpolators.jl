#= 

# 2D Commuting interpolators
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
    
    ## choose finite element spaces that allow commuting interpolators
    FETypeH1 = H1P1{1}
    FETypeHdiv = HDIVRT0{2}
    FES1 = FESpace{FETypeH1}(xgrid)
    FES2 = FESpace{FETypeHdiv}(xgrid)

    ## do the interpolations
    H1Interpolation = FEVector{Float64}("H1-Interpolation",FES1)
    HdivCurlInterpolation = FEVector{Float64}("Hdiv-Interpolation",FES2)
    interpolate!(H1Interpolation[1], exact_function!)
    interpolate!(HdivCurlInterpolation[1], exact_curl!; bonus_quadorder = 3)

    ## evaluate the nodal values of Curl(H1(psi)) and Hdiv(Curl(psi))
    ## and subtract from each other, all values should be close to zero
    nodevalsH1 = zeros(Float64,2,size(xgrid[Coordinates],2))
    nodevalues!(nodevalsH1,H1Interpolation[1],FES1,CurlScalar)
    nodevalsHdiv = zeros(Float64,2,size(xgrid[Coordinates],2))
    nodevalues!(nodevalsHdiv,HdivCurlInterpolation[1],FES2)
    println("\nnodevals(Curl(H1(psi) - Hdiv(Curl(psi))):")
    Base.show(nodevalsHdiv-nodevalsH1)
    
end

main()