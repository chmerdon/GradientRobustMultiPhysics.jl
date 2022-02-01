#= 

# A02 : Commuting Interpolators 2D
([source code](SOURCE_URL))

This example verifies a structural property of the H1 and Hdiv finite element spaces and their interpolators which is
```math
\mathrm{Curl}(I_{\mathrm{P}_k}\psi) = I_{\mathrm{RT}_{k-1}}(\mathrm{Curl}(\psi))
```
for the ``H_1`` interpolator ``I_{\mathrm{P}_k}`` and the standard Raviart-Thomas interpolator ``I_{\mathrm{RT}_{k-1}}`` for $k > 0$.
In this example we verify this identity for $k=1$ and $k=2$. Note, that the ``H_1`` interpolator only does nodal interpolations at the
vertices but not in the additional degrees of freedom. For ``k=2``, the interpolator also preserves the moments along the edges.

=#

module ExampleA02_CommutingInterpolators2D

using GradientRobustMultiPhysics
using ExtendableGrids

## define some function
function exact_function!(result,x)
    result[1] = x[1]^2-x[2]^4 + 1
end

## everything is wrapped in a main function
function main(;order::Int = 2, testmode = false)

    ## choose some grid
    xgrid = uniform_refine(reference_domain(Triangle2D),2)

    ## negotiate exact_function! and exact_curl! to the package
    u = DataFunction(exact_function!, [1,2]; name = "u_exact", dependencies = "X", bonus_quadorder = 4)
    u_curl = curl(u)

    ## choose commuting interpolators pair
    if order == 1
        FE = [H1P1{1},HDIVRT0{2}]; testFE = H1P0{2}
    elseif order == 2
        FE = [H1P2{1,2},HDIVRT1{2}]; testFE = H1P1{2}
    end

    ## do the H1 and Hdiv interpolation of the function and its curl, resp.
    FES = [FESpace{FE[1]}(xgrid), FESpace{FE[2]}(xgrid)]
    Interpolations = FEVector(["H1-Interpolation", "Hdiv-Interpolation"], FES)
    interpolate!(Interpolations[1], u)
    interpolate!(Interpolations[2], u_curl)

    ## Both sides of the identity are finite element functions of FEtype testFE
    ## Hence, we evaluate the error by testing the identity by all basisfunctions of this type
    
    ## Generate the test space and some matching FEVector
    FEStest = FESpace{testFE}(xgrid; broken = true)
    error = FEVector("ErrorVector",FEStest)

    ## Define (yet undiscrete) linear forms that represents testing each side of the identity with the testspace functions
    LF1 = LinearForm(Identity, [Identity], [1]) # identity of test function is multiplied with identity of other argument
    LF2 = LinearForm(Identity, [CurlScalar], [1]) # identity of test function is multiplied with CurlScalar of other argument

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
    for order in [1,2]
        push!(error, max(main(order = order, testmode = true)))
    end
    return maximum(error)
end

end
