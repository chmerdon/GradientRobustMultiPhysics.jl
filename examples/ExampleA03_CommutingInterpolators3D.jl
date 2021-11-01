#= 

# A03 : Commuting Interpolators 3D
([source code](SOURCE_URL))

This example verifies a structural property of the Hcurl and Hdiv finite element spaces and their interpolators which is
```math
\mathrm{Curl}(I_{\mathrm{N}_{k-1}}\psi) = I_{\mathrm{RT}_{k-1}}(\mathrm{Curl}(\psi))
```
for the standard Nedelec interpolator ``I_{\mathrm{N}_{k-1}}`` and the standard Raviart-Thomas interpolator ``I_{\mathrm{RT}_{k-1}}`` for $k > 0$.
In this example we verify this identity for $k=1$ (higher order spaces are tested as well as soon as they are available).
=#

module ExampleA03_CommutingInterpolators3D
using GradientRobustMultiPhysics
using ExtendableGrids

## define some function
function exact_function!(result,x)
    result[1] = x[2]^2 + x[3]
    result[2] = x[1]^3
    result[3] = 1 + x[3]^2
end
## and its Curl3D
function exact_curl!(result,x)
    result[1] = 0
    result[2] = 1
    result[3] = 3*x[1]^2 - 2*x[2]
end

## everything is wrapped in a main function
function main(;order::Int = 1, testmode = false)

    ## choose some grid
    xgrid = uniform_refine(reference_domain(Tetrahedron3D),2)

    ## negotiate exact_function! and exact_curl! to the package
    u = DataFunction(exact_function!, [3,3]; name = "u_exact", dependencies = "X", quadorder = 3)
    u_curl = DataFunction(exact_curl!, [3,3]; name = "Curl(u_exact)", dependencies = "X", quadorder = 2)

    ## choose commuting interpolators pair
    if order == 1
        FE = [HCURLN0{3},HDIVRT0{3}]; testFE = H1P0{3}
    end

    ## do the Hcurl interpolation of the function
    FESH1 = FESpace{FE[1]}(xgrid)
    HcurlInterpolation = FEVector("Hcurl-Interpolation",FESH1)
    interpolate!(HcurlInterpolation[1], u)

    ## do the Hdiv interpolation of the Curl of the function
    ## since integrals over faces have to be computed exactly we need to tune the quadrature order
    FESHdiv = FESpace{FE[2]}(xgrid)
    HdivCurlInterpolation = FEVector("Hdiv-Interpolation",FESHdiv)
    interpolate!(HdivCurlInterpolation[1], u_curl)

    ## Checking the identity:
    ## Both sides of the identity are finite element function of FEtype testFE
    ## Hence, we evaluate the error by testing the identity by all basisfunctions of this type
    
    ## first: generate the test space and some matching FEVector
    FEStest = FESpace{testFE}(xgrid; broken = true)
    error = FEVector("ErrorVector",FEStest)

    ## Define bilinear forms that represents testing each side of the identity with the testspace functions
    BLF1 = BilinearForm(Float64, ON_CELLS, [FEStest, FESHdiv], [Identity, Identity])
    BLF2 = BilinearForm(Float64, ON_CELLS, [FEStest, FESH1], [Identity, Curl3D])

    ## evaluate the bilinear forms in the respective interpolations and subtract them from each other
    ## note that in these calls always the second argument of the bilinearform is fixed by the given FEVectorBlock
    assemble!(error[1], BLF1, HdivCurlInterpolation[1]; fixed_arguments = [2])
    assemble!(error[1], BLF2, HcurlInterpolation[1]; factor = -1, fixed_arguments = [2])

    ## do some norm that recognizes a nonzero in the vector
    error = sqrt(sum(error[1][:].^2, dims = 1)[1])
    if testmode == true
        return error
    else
        println("error(Curl(I_$(FE[1])(psi) - I_$(FE[2])(Curl(psi))) = $error")
    end
end

## test function that is called by test unit
function test()
    error = []
    for order in [1]
        push!(error, max(main(order = order, testmode = true)))
    end
    return maximum(error)
end

end
