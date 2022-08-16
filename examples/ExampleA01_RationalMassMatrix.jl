#= 

# A01 : Rational Mass Matrix
([source code](SOURCE_URL))

This example demonstrates the usage of rational numbers to calculate e.g. exact mass matrices on reference domains
(if exact quadrature rules in Rational number format are available).

=#

module ExampleA01_RationalMassMatrix

using GradientRobustMultiPhysics
using ExtendableGrids

## everything is wrapped in a main function
function main()

    ## reference domain as extendable grid
    xgrid = reference_domain(Triangle2D, Rational{Int64})

    ## define P1-Courant finite element space
    FES = FESpace{H1P1{1}}(xgrid)

    ## define mass matrix bilinear form
    MAMA_BLF = DiscreteSymmetricBilinearForm([Identity,Identity],[FES,FES]; T = Rational{Int64})

    ## assemble mass matrix and divide by area
    MAMA = FEMatrix{Rational{Int64}}(FES)
    assemble!(MAMA[1],MAMA_BLF)
    MAMA = MAMA.entries ./ xgrid[CellVolumes][1]

    ## print matrix
    @show MAMA
end

end
