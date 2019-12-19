using Test
push!(LOAD_PATH, "../src")


using QuadratureTests
println("\nStarting tests for 1D Quadrature")
for order = 1 : 2
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order,1)
end
println("\nStarting tests for 2D Quadrature")
for order = 1 : 20
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order,2)
end


using FiniteElementsTests
println("\nStarting tests for FiniteElements")
@test FiniteElementsTests.TestP0()
@test FiniteElementsTests.TestP1()
@test FiniteElementsTests.TestP1V()
@test FiniteElementsTests.TestP2()
@test FiniteElementsTests.TestP2V()
@test FiniteElementsTests.TestBR()
@test FiniteElementsTests.TestCR()


using FESolveCommonTests
println("\nStarting tests for FESolveCommon")
@test FESolveCommonTests.TestInterpolation1D()
@test FESolveCommonTests.TestL2BestApproximation1D()
@test FESolveCommonTests.TestH1BestApproximation1D()
@test FESolveCommonTests.TestInterpolation2D()
@test FESolveCommonTests.TestL2BestApproximation2DP1()
@test FESolveCommonTests.TestL2BestApproximation2DCR()
@test FESolveCommonTests.TestL2BestApproximation2DP2()
@test FESolveCommonTests.TestH1BestApproximation2D()

using FESolvePoissonTests
println("\nStarting tests for FESolvePoisson")
@test FESolvePoissonTests.TestPoissonSolver1D()
@test FESolvePoissonTests.TestPoissonSolver2DP1()
@test FESolvePoissonTests.TestPoissonSolver2DCR()
@test FESolvePoissonTests.TestPoissonSolver2DP2()


using FESolveStokesTests
println("\nStarting tests for FESolveStokes")
@test FESolveStokesTests.TestStokesTH()
