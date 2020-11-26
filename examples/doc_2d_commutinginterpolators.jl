#= 

# Commuting Interpolators (2D)
([source code](SOURCE_URL))

This example verifies a structural property of the H1 and Hdiv finite element spaces and their interpolators which is
```math
\mathrm{Curl}(I_{\mathrm{P}_k}\psi) = I_{\mathrm{RT}_{k-1}}(\mathrm{Curl}(\psi))
```
for the ``H_1`` interpolator ``I_{\mathrm{P}_k}`` and the standard Raviart-Thomas interpolator ``I_{\mathrm{RT}_{k-1}}`` for $k > 0$.
In this example we verify this identity for $k=1$ and $k=2$. Note, that the ``H_1`` interpolator only does nodal interpolations at the
vertices but not in the additional edgrees of freedom. For e.g. ``k=2`` the interpolator preserves the moments along the edges.

!!! note

    In 3D a similar commuting property holds that involves the Nedelec finite element spaces,
    that will be tested once they are implemented. Also the identities for $k > 2$ will be tested once all functionality is available.

=#

module Example_2DCommutingInterpolators
using GradientRobustMultiPhysics

## define some function
function exact_function!(result,x)
    result[1] = x[1]^2-x[2]^4 + 1
end
## and its CurlScalar = (-dy,dx) 
function exact_curl!(result,x)
    result[1] = 4*x[2]^3
    result[2] = 2*x[1]
end

## everything is wrapped in a main function
function main(;order::Int = 2)

    ## choose some grid
    xgrid = uniform_refine(reference_domain(Triangle2D),2)

    ## choose commuting interpolators pair
    if order == 1
        FE = [H1P1{1},HDIVRT0{2}]; testFE = L2P0{2}
    elseif order == 2
        FE = [H1P2{1,2},HDIVRT1{2}]; testFE = L2P1{2}
    end

    ## do the H1 interpolation of the function
    FESH1 = FESpace{FE[1]}(xgrid)
    H1Interpolation = FEVector{Float64}("H1-Interpolation",FESH1)
    interpolate!(H1Interpolation[1], exact_function!; bonus_quadorder = 4)

    ## do the Hdiv interpolation of the Curl of the function
    ## since integrals over faces have to be computed exactly we need to tune the quadrature order
    FESHdiv = FESpace{FE[2]}(xgrid)
    HdivCurlInterpolation = FEVector{Float64}("Hdiv-Interpolation",FESHdiv)
    interpolate!(HdivCurlInterpolation[1], exact_curl!; bonus_quadorder = 3)

    ## Checking the identity:
    ## Both sides of the identity are finite element function of FEtype testFE
    ## Hence, we evaluate the error by testing the identity by all basisfunctions of this type
    
    ## first: generate the test space and some matching FEVector
    FEStest = FESpace{testFE}(xgrid)
    error = FEVector{Float64}("ErrorVector",FEStest)

    ## Define bilinear forms that represents testing each side of the identity with the testspace functions
    BLF1 = BilinearForm(Float64, ON_CELLS, FEStest, FESHdiv, Identity, Identity, DoNotChangeAction(2))
    BLF2 = BilinearForm(Float64, ON_CELLS, FEStest, FESH1, Identity, CurlScalar, DoNotChangeAction(2))

    ## evaluate the bilinear forms in the respective interpolations and subtract them from each other
    ## note that in these calls always the second argument of the bilinearform is fixed by the given FEVectorBlock
    assemble!(error[1], HdivCurlInterpolation[1], BLF1)
    assemble!(error[1], H1Interpolation[1], BLF2; factor = -1)

    ## do some norm that recognizes a nonzero in the vector
    error = sqrt(sum(error[1][:].^2, dims = 1)[1])
    println("error(Curl(I_$(FE[1])(psi) - I_$(FE[2])(Curl(psi))) = $error")
    return error
end

## test function that is called by test unit
function test()
    error = []
    for order in [1,2]
        push!(error, max(main(order = order)))
    end
    return maximum(error)
end

end