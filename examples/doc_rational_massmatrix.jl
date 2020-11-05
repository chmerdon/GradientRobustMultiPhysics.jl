#= 

# Ration Mass Matrix
([source code](SOURCE_URL))

This example demonstrates the usage of rational numbers to calculate e.g. exact mass matrices on reference domains
(if exact quadrature rules in Rational number format are available).

=#

module Example_RationalMAMA

using GradientRobustMultiPhysics
using ExtendableGrids


## everything is wrapped in a main function
function main(; verbosity = 1, Plotter = nothing)

    ## reference domain as extendable grid
    xgrid = reference_domain(Triangle2D, Rational)

    ## define finite element space
    FEType = H1P1{1}
    FES = FESpace{FEType}(xgrid)

    ## define mass matrix bilinear form
    MAMA_BLF = SymmetricBilinearForm(Rational,ON_CELLS,FES,Identity,DoNotChangeAction(1))

    ## assemble mass matrix
    MAMA = FEMatrix{Rational}("mass matrix for $(FEType)",FES)
    assemble!(MAMA[1],MAMA_BLF; verbosity = verbosity)

    ## show mass matrix divided by area
    Base.show(MAMA.entries./xgrid[CellVolumes][1])

end

end