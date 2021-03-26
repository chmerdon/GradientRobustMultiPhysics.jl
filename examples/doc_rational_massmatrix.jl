#= 

# Rational Mass Matrix
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

    ## define P1-Courant finite element space
    FES = FESpace{H1P1{1}}(xgrid)

    ## define mass matrix bilinear form
    MAMA_BLF = SymmetricBilinearForm(Rational,ON_CELLS,[FES,FES],[Identity,Identity],DoNotChangeAction(1,Rational))

    ## assemble mass matrix and divide by area
    MAMA = FEMatrix{Rational}("mass matrix",FES)
    assemble!(MAMA[1],MAMA_BLF)
    MAMA = MAMA.entries ./ xgrid[CellVolumes][1]

    ## print matrix
    @show MAMA
end

end

#=
### Output of default main() run
=#
Example_RationalMAMA.main()