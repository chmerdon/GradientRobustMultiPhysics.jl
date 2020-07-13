using Test
using ExtendableGrids
using LinearAlgebra

push!(LOAD_PATH, "../src")

############################
# TESTSET QUADRATURE RULES #
############################

using QuadratureRules
using FEXGrid

include("../src/testgrids.jl")

function exact_function(polyorder)
    function closure(result,x)
        result[1] = x[1]^polyorder + 2*x[2]^polyorder + 1
        result[2] = 3*x[1]^polyorder - x[2]^polyorder - 1
    end
    exact_integral = [3 // (polyorder+1) + 1, 2 // (polyorder+1) - 1]
    return closure, exact_integral
end


@testset "QuadratureRules on Triangles/Quads" begin
    println("\n")
    println("==========================================")
    println("Testing QuadratureRules on Triangles/Quads")
    println("==========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for order = 1 : 10
        integrand!, exactvalue = exact_function(order)
        quadvalue = integrate!(xgrid, AbstractAssemblyTypeCELL, integrand!, order, length(exactvalue))
        println("order = $order | error = $(quadvalue - exactvalue)")
        @test isapprox(quadvalue,exactvalue)
    end
    println("")
end


#################################
# TESTSET FE-BESTAPPROXIMATIONS #
#################################

using FiniteElements
using FEAssembly
using PDETools



# list of FETypes that should be tested
TestCatalog = [FiniteElements.HDIVRT0{2},
               FiniteElements.L2P0{2},
               FiniteElements.H1P1{2}, 
               FiniteElements.H1CR{2},
               FiniteElements.H1MINI{2,2},
               FiniteElements.H1BR{2},
               FiniteElements.L2P1{2},
               FiniteElements.H1P2{2,2}]
ExpectedOrders = [0,0,1,1,1,1,1,2]

@testset "L2-Bestapprox on Triangles/Quads" begin
    println("\n")
    println("================================================")
    println("Testing L2-Bestapproximations on Triangles/Quads")
    println("================================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog)
        exact_function!, exactvalue = exact_function(ExpectedOrders[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        Problem = L2BestapproximationProblem(exact_function!, 2, length(exactvalue); bestapprox_boundary_regions = [], bonus_quadorder = ExpectedOrders[n])
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, length(exactvalue); bonus_quadorder = ExpectedOrders[n])

        # choose FE and generate FESpace
        FEType = TestCatalog[n]
        FESpace = FiniteElements.FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("L2-Bestapproximation",FESpace)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end
