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
fems1D = ["P1","P2"]
expected_orders1D = [1,2]
for j = 1 : length(fems1D)
    for d = 1 : 2
        @test FESolveCommonTests.TestInterpolation1D(fems1D[j],expected_orders1D[j],d)
    end    
    @test FESolveCommonTests.TestBestApproximation1D("L2",fems1D[j],expected_orders1D[j])
    @test FESolveCommonTests.TestBestApproximation1D("H1",fems1D[j],expected_orders1D[j])
end   
fems2D = ["P1","P2","CR","MINI","P2B"]
expected_orders2D = [1,2,1,1,2]
for j = 1 : length(fems2D)
    for d = 1 : 2
        @test FESolveCommonTests.TestInterpolation2D(fems2D[j],expected_orders2D[j],d)
    end    
    @test FESolveCommonTests.TestBestApproximation2D("L2",fems2D[j],expected_orders2D[j])
    @test FESolveCommonTests.TestBestApproximation2D("H1",fems2D[j],expected_orders2D[j])
end    
fems2D_squares = ["Q1"]
expected_orders2D = [1,2,1,1,2]
for j = 1 : length(fems2D_squares)
    for d = 1 : 2
        @test FESolveCommonTests.TestInterpolation2D(fems2D_squares[j],expected_orders2D[j],d,true)
    end    
    @test FESolveCommonTests.TestBestApproximation2D("L2",fems2D_squares[j],expected_orders2D[j], true)
    @test FESolveCommonTests.TestBestApproximation2D("H1",fems2D_squares[j],expected_orders2D[j], true)
end    
# test BR interpolation on squares
@test FESolveCommonTests.TestInterpolation2D("BR",1,2,true)

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
fems2D_squares = ["Q1"]
expected_orders2D = [1,2,1,1,2]
for j = 1 : length(fems2D_squares)
    @test FESolvePoissonTests.TestPoissonSolver2D(fems2D_squares[j],expected_orders2D[j],true)
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
fems_velocity2D = ["MINI","P2","BR","CR","P2B"]
fems_pressure2D = ["P1","P1","P0","P0","P1dc"]
expected_orders2D = [1,2,1,1,2]
for j = 1 : length(fems_velocity2D)
    @test FESolveNavierStokesTests.TestNavierStokesSolver2D(fems_velocity2D[j],fems_pressure2D[j],expected_orders2D[j])
end   
