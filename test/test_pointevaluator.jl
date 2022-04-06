

function run_pointevaluator_tests()
    @testset "Operators" begin
        println("\n")
        println("=========================")
        println("Testing Point Evaluations")
        println("=========================")
        error = test_pointevaluation()
        @test error < 1e-14
    end
end

function test_pointevaluation()
    ## check if a linear function is interpolated correctly
    ## and can be evaluated correctly within the cell
    xgrid = grid_triangle([-1.0 0.0; 1.0 0.0; 0.0 1.0]')
    FES = FESpace{H1P1{1}}(xgrid)
    Iu = FEVector("Iu",FES)
    interpolate!(Iu[1], DataFunction((result,x) -> (result[1] = x[1] + x[2];), [1,2]; dependencies = "X"))
    PE = PointEvaluator(Iu[1], Identity)
    CF = CellFinder(xgrid)
    eval = zeros(Float64,1)
    x = [0.15234,0.2234] # point inside the cell
    xref = [0.0,0.0]
    cell = gFindLocal!(xref, CF, x; icellstart = 1)
    evaluate!(eval,PE,xref,cell)
    return abs(eval[1] - sum(x))
end