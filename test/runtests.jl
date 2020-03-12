using Test
push!(LOAD_PATH, "../src")


using QuadratureTests
println("\nStarting tests for 1D Quadrature")
for order = 1 : 10
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order,1)
end
println("\nStarting tests for 2D Quadrature")
for order = 1 : 10
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order,2)
end

using FESolveCommonTests
println("\nStarting tests for FESolveCommon")
#@test FESolveCommonTests.TestInterpolation1D()
#@test FESolveCommonTests.TestL2BestApproximation1D()
#@test FESolveCommonTests.TestH1BestApproximation1D()
#@test FESolveCommonTests.TestInterpolation2D()
#@test FESolveCommonTests.TestL2BestApproximation2DP1()
#@test FESolveCommonTests.TestL2BestApproximation2DCR()
#@test FESolveCommonTests.TestL2BestApproximation2DP2()
#@test FESolveCommonTests.TestH1BestApproximation2D()

using FESolvePoissonTests
println("\nStarting tests for FESolvePoisson")
fems1D = ["P1","P2"]
expected_orders1D = [1,2]
for j = 1 : length(fems1D)
    @test FESolvePoissonTests.TestPoissonSolver1D(fems1D[j],expected_orders1D[j])
end    
fems2D = ["P1","P2","CR","MINI","P2B"]
expected_orders2D = [1,2,1,1,2]
for j = 1 : length(fems2D)
    @test FESolvePoissonTests.TestPoissonSolver2D(fems2D[j],expected_orders2D[j])
end    


using FESolveStokesTests
println("\nStarting tests for FESolveStokes")  
fems_velocity2D = ["MINI","P2","CR","BR","P2B"]
fems_pressure2D = ["P1","P1","P0","P0","P1dc"]
expected_orders2D = [1,2,1,1,2]
for j = 1 : length(fems_velocity2D)
    @test FESolveStokesTests.TestStokesSolver2D(fems_velocity2D[j],fems_pressure2D[j],expected_orders2D[j])
end   


using FESolveNavierStokesTests
println("\nStarting tests for FESolveNavierStokes")  
fems_velocity2D = ["MINI","P2","BR","P2B"]
fems_pressure2D = ["P1","P1","P0","P1dc"]
# CR/P0 not included yet
expected_orders2D = [1,2,1,2]
for j = 1 : length(fems_velocity2D)
    @test FESolveNavierStokesTests.TestNavierStokesSolver2D(fems_velocity2D[j],fems_pressure2D[j],expected_orders2D[j])
end   
