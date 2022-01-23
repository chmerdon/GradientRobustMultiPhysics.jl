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

## everything is wrapped in a main function
function main(;order::Int = 1, testmode = false)

    ## choose some grid
    xgrid = uniform_refine(reference_domain(Tetrahedron3D),2)

    ## negotiate exact_function! and exact_curl! to the package
    u = DataFunction(exact_function!, [3,3]; name = "u_exact", dependencies = "X", quadorder = 3)
    u_curl = curl(u)

    ## choose commuting interpolators pair
    if order == 1
        FE = [HCURLN0{3},HDIVRT0{3}]; testFE = H1P0{3}
    end

    ## do the Hcurl and Hdiv interpolation of the function and its curl, resp.
    FES = [FESpace{FE[1]}(xgrid), FESpace{FE[2]}(xgrid)]
    Interpolations = FEVector(["Hcurl-Interpolation", "Hdiv-Interpolation"], FES)
    interpolate!(Interpolations[1], u)
    interpolate!(Interpolations[2], u_curl)

    ## Both sides of the identity are finite element functions of FEtype testFE
    ## Hence, we evaluate the error by testing the identity by all basisfunctions of this type
    
    ## Generate the test space and some matching FEVector
    FEStest = FESpace{testFE}(xgrid; broken = true)
    error = FEVector("ErrorVector",FEStest)

    ## Define (yet undiscrete) linear forms that represents testing each side of the identity with the testspace functions
    LF1 = LinearForm(Identity, [Identity]) # identity of test function is multiplied with identity of other argument
    LF2 = LinearForm(Identity, [Curl3D]) # identity of test function is multiplied with Curl3D of other argument

    ## Assemble linear forms into the same vector with opposite signs
    ## note: first argument fixes the test function FESpace and third arguments are used for the additional operators of the linearform
    assemble_operator!(error[1], LF1, [Interpolations[2]])
    assemble_operator!(error[1], LF2, [Interpolations[1]]; factor = -1)

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