using Test
using ExtendableGrids

push!(LOAD_PATH, "../src")
using GradientRobustMultiPhysics


include("../src/testgrids.jl")

function exact_function1D(polyorder)
    function closure(result,x)
        result[1] = x[1]^polyorder + 1
    end
    function gradient(result,x)
        result[1] = polyorder * x[1]^(polyorder-1)
    end
    exact_integral = [1 // (polyorder+1) + 1]
    return closure, exact_integral, gradient
end

function exact_function2D(polyorder)
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

############################
# TESTSET QUADRATURE RULES #
############################
maxorder1D = 8
maxorder2D = 16

@testset "QuadratureRules on Triangles/Quads" begin
    println("\n")
    println("====================================")
    println("Testing QuadratureRules on Intervals")
    println("====================================")
    xgrid = simplexgrid([0.0,1//4,2//3,1.0]); # initial grid
    for order = 1 : maxorder1D
        integrand!, exactvalue = exact_function1D(order)
        quadvalue = integrate(xgrid, AssemblyTypeCELL, integrand!, order, length(exactvalue))
        println("order = $order | error = $(quadvalue - exactvalue)")
        @test isapprox(quadvalue,exactvalue)
    end
    println("\n")
    println("==========================================")
    println("Testing QuadratureRules on Triangles/Quads")
    println("==========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for order = 1 : maxorder2D
        integrand!, exactvalue = exact_function2D(order)
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
TestCatalog1D = [
                L2P0{1},
                H1P1{1}, 
                L2P1{1},
                H1P2{1,1}]
ExpectedOrders1D = [0,1,1,2]
TestCatalog2D = [
#                HDIVRT0{2}, # has not interpolation yet
                L2P0{2},
                H1P1{2}, 
                H1CR{2},
                H1MINI{2,2},
                H1BR{2},
                L2P1{2},
                H1P2{2,2}]
ExpectedOrders2D = [0,1,1,1,1,1,2]


@testset "Interpolations" begin
    println("\n")
    println("===================================")
    println("Testing Interpolations on Intervals")
    println("===================================")
    xgrid = simplexgrid([0.0,1//4,2//3,1.0]); # initial grid
    for n = 1 : length(TestCatalog1D)
        exact_function!, exactvalue = exact_function1D(ExpectedOrders1D[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 1, length(exactvalue); bonus_quadorder = ExpectedOrders1D[n])

        # choose FE and generate FESpace
        FEType = TestCatalog1D[n]
        FES = FESpace{FEType}(xgrid)

        # interpolate
        Solution = FEVector{Float64}("Interpolation",FES)
        interpolate!(Solution[1], exact_function!; bonus_quadorder = ExpectedOrders1D[n])

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders1D[n]) | error = $error")
        @test error < 1e-12
    end
    println("\n")
    println("=========================================")
    println("Testing Interpolations on Triangles/Quads")
    println("=========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog2D)
        exact_function!, exactvalue = exact_function2D(ExpectedOrders2D[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, length(exactvalue); bonus_quadorder = ExpectedOrders2D[n])

        # choose FE and generate FESpace
        FEType = TestCatalog2D[n]
        FES = FESpace{FEType}(xgrid)

        # interpolate
        Solution = FEVector{Float64}("Interpolation",FES)
        interpolate!(Solution[1], exact_function!; bonus_quadorder = ExpectedOrders2D[n])

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders2D[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end

#################################################
# TESTSET Finite Elements L2-Bestapproximations #
#################################################

# list of FETypes that should be tested
TestCatalog1D = [
                L2P0{1},
                H1P1{1}, 
                L2P1{1},
                H1P2{1,1}]
ExpectedOrders1D = [0,1,1,2]
TestCatalog2D = [
                HDIVRT0{2},
                L2P0{2},
                H1P1{2}, 
                H1CR{2},
                H1MINI{2,2},
                H1BR{2},
                L2P1{2},
                H1P2{2,2}]
ExpectedOrders2D = [0,0,1,1,1,1,1,2]

@testset "L2-Bestapproximations" begin
    println("\n")
    println("==========================================")
    println("Testing L2-Bestapproximations on Intervals")
    println("==========================================")
    xgrid = simplexgrid([0.0,1//4,2//3,1.0]); # initial grid
    for n = 1 : length(TestCatalog1D)
        exact_function!, exactvalue = exact_function1D(ExpectedOrders1D[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        Problem = L2BestapproximationProblem(exact_function!, 1, length(exactvalue); bestapprox_boundary_regions = [], bonus_quadorder = ExpectedOrders1D[n])
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 1, length(exactvalue); bonus_quadorder = ExpectedOrders1D[n])

        # choose FE and generate FESpace
        FEType = TestCatalog1D[n]
        FES = FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("L2-Bestapproximation",FES)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders1D[n]) | error = $error")
        @test error < 1e-12
    end
    println("\n")
    println("================================================")
    println("Testing L2-Bestapproximations on Triangles/Quads")
    println("================================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog2D)
        exact_function!, exactvalue = exact_function2D(ExpectedOrders2D[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        Problem = L2BestapproximationProblem(exact_function!, 2, length(exactvalue); bestapprox_boundary_regions = [], bonus_quadorder = ExpectedOrders2D[n])
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, length(exactvalue); bonus_quadorder = ExpectedOrders2D[n])

        # choose FE and generate FESpace
        FEType = TestCatalog2D[n]
        FES = FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("L2-Bestapproximation",FES)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders2D[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end


#################################################
# TESTSET Finite Elements H1-Bestapproximations #
#################################################

# list of FETypes that should be tested
TestCatalog1D = [
                H1P1{1}, 
                H1P2{1,1}]
ExpectedOrders1D = [1,2]
TestCatalog2D = [
                H1P1{2}, 
                H1CR{2},
                H1MINI{2,2},
                H1BR{2},
                H1P2{2,2}]
ExpectedOrders2D = [1,1,1,1,2]

@testset "H1-Bestapproximations" begin
    println("\n")
    println("==========================================")
    println("Testing H1-Bestapproximations on Intervals")
    println("==========================================")
    xgrid = simplexgrid([0.0,1//4,2//3,1.0]); # initial grid
    for n = 1 : length(TestCatalog1D)
        exact_function!, exactvalue, exact_function_gradient! = exact_function1D(ExpectedOrders1D[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        Problem = H1BestapproximationProblem(exact_function_gradient!, exact_function!, 1, length(exactvalue); bestapprox_boundary_regions = [1,2], bonus_quadorder = ExpectedOrders1D[n] - 1, bonus_quadorder_boundary = ExpectedOrders1D[n])
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 1, length(exactvalue); bonus_quadorder = ExpectedOrders1D[n])

        # choose FE and generate FESpace
        FEType = TestCatalog1D[n]
        FES = FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("H1-Bestapproximation",FES)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders1D[n]) | error = $error")
        @test error < 1e-12
    end
    println("\n")
    println("================================================")
    println("Testing H1-Bestapproximations on Triangles/Quads")
    println("================================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog2D)
        exact_function!, exactvalue, exact_function_gradient! = exact_function2D(ExpectedOrders2D[n])

        # Define Bestapproximation problem via PDETooles_PDEProtoTypes
        Problem = H1BestapproximationProblem(exact_function_gradient!, exact_function!, 2, length(exactvalue); bestapprox_boundary_regions = [1,2,3,4], bonus_quadorder = ExpectedOrders2D[n] - 1, bonus_quadorder_boundary = ExpectedOrders2D[n])
        L2ErrorEvaluator = L2ErrorIntegrator(exact_function!, Identity, 2, length(exactvalue); bonus_quadorder = ExpectedOrders2D[n])

        # choose FE and generate FESpace
        FEType = TestCatalog2D[n]
        FES = FESpace{FEType}(xgrid)

        # solve
        Solution = FEVector{Float64}("H1-Bestapproximation",FES)
        solve!(Solution, Problem)

        # check error
        error = sqrt(evaluate(L2ErrorEvaluator,Solution[1]))
        println("FEType = $FEType | order = $(ExpectedOrders2D[n]) | error = $error")
        @test error < 1e-12
    end
    println("")
end



###########################
# TESTSET Stokes elements #
###########################


function exact_functions_stokes2D(polyorder_velocity,polyorder_pressure)
    function exact_velocity!(result,x)
        result[1] = x[2]^polyorder_velocity + 1
        result[2] = x[1]^polyorder_velocity - 1
    end
    function exact_pressure!(result,x)
        result[1] = x[1]^polyorder_pressure + x[2]^polyorder_pressure - 2 // (polyorder_pressure+1)
    end
    function velo_gradient!(result,x)
        result[1] = 0
        result[2] = polyorder_velocity * x[2]^(polyorder_velocity-1)
        result[3] = polyorder_velocity * x[1]^(polyorder_velocity-1)
        result[4] = 0
    end
    function rhs!(result,x) # = Delta u + grad p
        result[1] = 0
        result[2] = 0
        if polyorder_velocity > 1
            result[1] -= polyorder_velocity*(polyorder_velocity-1)*x[2]^(polyorder_velocity-2)
            result[2] -= polyorder_velocity*(polyorder_velocity-1)*x[1]^(polyorder_velocity-2)
        end
        if polyorder_pressure > 0
            result[1] += polyorder_pressure * x[1]^(polyorder_pressure-1)
            result[2] += polyorder_pressure * x[2]^(polyorder_pressure-1)
        end
    end
    return exact_velocity!, exact_pressure!, velo_gradient!, rhs!
end

# list of FETypes that should be tested
TestCatalog2D = [
                [H1CR{2},L2P0{1}],
                [H1MINI{2,2},H1CR{1}], # taking here a stable variant that also works on quads
                [H1BR{2},L2P0{1}],
                [H1P2{2,2},H1P1{1}]]
ExpectedOrders2D = [[1,0],[1,1],[1,0],[2,1]]

@testset "Stokes-FEM" begin
    println("\n")
    println("==========================================")
    println("Testing Stokes elements on Triangles/Quads")
    println("==========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for n = 1 : length(TestCatalog2D)
        exact_velocity!, exact_pressure!, exact_function_gradient!, rhs! = exact_functions_stokes2D(ExpectedOrders2D[n][1],ExpectedOrders2D[n][2])

        # Define Stokes problem via PDETooles_PDEProtoTypes
        StokesProblem = IncompressibleNavierStokesProblem(2; nonlinear = false)
        add_boundarydata!(StokesProblem, 1, [1,2,3,4], BestapproxDirichletBoundary; data = exact_velocity!, bonus_quadorder = ExpectedOrders2D[n][1])
        add_rhsdata!(StokesProblem, 1, RhsOperator(Identity, [rhs!], 2, 2; bonus_quadorder = max(0,ExpectedOrders2D[n][2]-1)))
        L2ErrorEvaluatorV = L2ErrorIntegrator(exact_velocity!, Identity, 2, 2; bonus_quadorder = ExpectedOrders2D[n][1])
        L2ErrorEvaluatorP = L2ErrorIntegrator(exact_pressure!, Identity, 2, 1; bonus_quadorder = ExpectedOrders2D[n][2])

        # choose FE and generate FESpace
        FETypes = TestCatalog2D[n]
        FES = [FESpace{FETypes[1]}(xgrid),FESpace{FETypes[2]}(xgrid)]

        # solve
        Solution = FEVector{Float64}("H1-Bestapproximation",FES)
        solve!(Solution, StokesProblem)

        # check error
        errorV = sqrt(evaluate(L2ErrorEvaluatorV,Solution[1]))
        errorP = sqrt(evaluate(L2ErrorEvaluatorP,Solution[2]))
        println("FETypes = $FETypes | orders = $(ExpectedOrders2D[n]) | errorV = $errorV | errorP = $errorP")
        @test max(errorV,errorP) < 1e-12
    end
    println("")
end
