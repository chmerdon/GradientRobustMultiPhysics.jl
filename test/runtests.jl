using Test
using ExtendableGrids

push!(LOAD_PATH, "../src")
using JUFELIA

############################
# TESTSET QUADRATURE RULES #
############################

include("../src/testgrids.jl")

function exact_function(polyorder)
    function closure(result,x)
        result[1] = x[1]^polyorder + 2*x[2]^polyorder + 1
        result[2] = 3*x[1]^polyorder - x[2]^polyorder - 1
    end
    function gradient(result,x)
        result[1] = polyorder * x[1]^(polyorder-1)
        result[2] = 2 * polyorder * x[2]^(polyorder - 1)
        result[3] = 3 * polyorder * x[1]^(polyorder-1)
        result[4] = - polyorder * x[2]^(polyorder - 1)
    end
    exact_integral = [3 // (polyorder+1) + 1, 2 // (polyorder+1) - 1]
    return closure, exact_integral, gradient
end


@testset "QuadratureRules on Triangles/Quads" begin
    println("\n")
    println("==========================================")
    println("Testing QuadratureRules on Triangles/Quads")
    println("==========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for order = 1 : 10
        integrand!, exactvalue = exact_function(order)
        quadvalue = integrate(xgrid, AssemblyTypeCELL, integrand!, order, length(exactvalue))
        println("order = $order | error = $(quadvalue - exactvalue)")
        @test isapprox(quadvalue,exactvalue)
    end
    println("")
end


##########################################
# TESTSET Finite Elements interpolations #
##########################################

# list of FETypes that should be tested
TestCatalog = [
#                HDIVRT0{2},
                L2P0{2},
                H1P1{2}, 
                H1CR{2},
                H1MINI{2,2},
                H1BR{2},
                L2P1{2},
                H1P2{2,2}]
ExpectedOrders = [0,1,1,1,1,1,2]


@testset "Interpolations on Triangles/Quads" begin
    println("\n")
    println("=========================================")
    println("Testing Interpolations on Triangles/Quads")
    println("=========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog)
        exact_function!, exactvalue = exact_function(ExpectedOrders[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, length(exactvalue); bonus_quadorder = ExpectedOrders[n])

        # choose FE and generate FESpace
        FEType = TestCatalog[n]
        FES = FESpace{FEType}(xgrid)

        # interpolate
        Solution = FEVector{Float64}("L2-Bestapproximation",FES)
        interpolate!(Solution[1], exact_function!; bonus_quadorder = ExpectedOrders[n])

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end

#################################################
# TESTSET Finite Elements L2-Bestapproximations #
#################################################

# list of FETypes that should be tested
TestCatalog = [HDIVRT0{2},
                L2P0{2},
                H1P1{2}, 
                H1CR{2},
                H1MINI{2,2},
                H1BR{2},
                L2P1{2},
                H1P2{2,2}]
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
        FES = FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("L2-Bestapproximation",FES)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end

# list of FETypes that should be tested
TestCatalog = [H1P1{2}, 
                H1CR{2},
                H1MINI{2,2},
                H1BR{2},
                H1P2{2,2}]
ExpectedOrders = [1,1,1,1,2]

@testset "H1-Bestapprox on Triangles/Quads" begin
    println("\n")
    println("================================================")
    println("Testing H1-Bestapproximations on Triangles/Quads")
    println("================================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog)
        exact_function!, exactvalue, exact_function_gradient! = exact_function(ExpectedOrders[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        Problem = H1BestapproximationProblem(exact_function_gradient!, exact_function!, 2, length(exactvalue); bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = ExpectedOrders[n] - 1, bonus_quadorder_boundary = ExpectedOrders[n])
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, length(exactvalue); bonus_quadorder = ExpectedOrders[n])

        # choose FE and generate FESpace
        FEType = TestCatalog[n]
        FES = FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("H1-Bestapproximation",FES)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end
