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
function main(; verbosity = 0, Plotter = nothing)

    ## reference domain as extendable grid
    xgrid = reference_domain(Triangle2D, Rational)

    ## define finite element space
    FEType = H1P1{1}
    FES = FESpace{FEType}(xgrid)

    ## define mass matrix bilinear form
    MAMA_BLF = SymmetricBilinearForm(Rational,ON_CELLS,[FES,FES],[Identity,Identity],DoNotChangeAction(1))

    ## assemble mass matrix
    MAMA = FEMatrix{Rational}("mass matrix for $(FEType)",FES)
    assemble!(MAMA[1],MAMA_BLF; verbosity = verbosity)

    ## divide by area
    MAMA.entries ./= xgrid[CellVolumes][1]

    ## print matrix
    for j = 1 : size(MAMA.entries,1)
        println("$(MAMA.entries[j,:])")
    end
end

end

#=
### Output of default main() run
=#
Example_RationalMAMA.main()