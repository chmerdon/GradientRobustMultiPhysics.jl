using Test
using ExtendableGrids
using LinearAlgebra

push!(LOAD_PATH, "../src")
using QuadratureRules
using FEXGrid

include("../src/testgrids.jl")


function integrand(polyorder)
    function closure(result,x)
        result[1] = x[1]^polyorder + 2*x[2]^polyorder + 1
        result[2] = 3*x[1]^polyorder - x[2]^polyorder - 1
    end
    exact_integral = [3 // (polyorder+1) + 1, 2 // (polyorder+1) - 1]
    return closure, exact_integral
end

@testset "QuadratureRules on Triangles/Quads" begin
    println("Testing QuadratureRules on Triangles/Quads")
    println("==========================================")
    xgrid = testgrid_mixedEG(); # initial grid
    for order = 1 : 10
        integrand!, exactvalue = integrand(order)
        quadvalue = integrate!(xgrid, AbstractAssemblyTypeCELL, integrand!, order, length(exactvalue))
        println("order = $order | error = $(quadvalue - exactvalue)")
        @test isapprox(quadvalue,exactvalue)
    end
end;