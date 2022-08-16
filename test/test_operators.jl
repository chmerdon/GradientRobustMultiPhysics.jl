
function run_operator_tests()
    @testset "Operators" begin
        println("\n")
        println("============================")
        println("Testing Operator Evaluations")
        println("============================")
        error = test_2nd_derivs()
        @test error < 1e-14
        error = test_recastBLFintoLF()
        @test error < 1e-14
    end
end

function test_recastBLFintoLF()
    ## define grid = a single non-refenrece triangle
    xgrid = grid_triangle([-1.0 0.0; 1.0 0.0; 0.0 1.0]')

    ## define P2-Courant finite element space
    FETypes = [H1P2{2,2}, H1P1{1}]
    FES = [FESpace{FETypes[1]}(xgrid),FESpace{FETypes[2]}(xgrid; broken = true)]
    
    ## test recasting of Bilinearform into LinearForm
    BLF = BilinearForm([Divergence, Identity]; name = "(div(v),p)")
    LF1 = LinearForm(Divergence, [Identity], [2]; name = "(div(v),p)")    # LinearForm for argument 1
    LF2 = LinearForm(Identity, [Divergence], [1]; name = "(p,div(v))")    # LinearForm for argument 2
    
    ## assemble BLF into matrix
    B = FEMatrix{Float64}(FES[1], FES[2])
    assemble_operator!(B[1,1], BLF)

    ## generate test vector with some constant functions
    Test = FEVector(FES)
    interpolate!(Test[1], DataFunction((result,x) -> (result[1] = x[1]; result[2] = x[2]), [2, 2]; dependencies = "X", bonus_quadorder = 1)) # div(u) = 2
    interpolate!(Test[2], DataFunction([0.75]))

    ## assemble LF1 and LF2 into vectors
    b1 = FEVector(FES[1])
    b2 = FEVector(FES[2])
    assemble_operator!(b1[1], LF1, Test)
    assemble_operator!(b2[1], LF2, Test)

    ## check if results are as expected (div(u)*p) = 2*0.75 = 1.5
    error_B = (lrmatmul(Test[1], B.entries, Test[2]) - 1.5)^2
    error_b1 = (dot(b1[1],Test[1]) - 1.5)^2
    error_b2 = (dot(b2[1],Test[2]) - 1.5)^2

    ## subtract matrix-vector product (should be the same)
    addblock_matmul!(b1[1], B[1,1], Test[2]; factor = -1)
    addblock_matmul!(b2[1], B[1,1], Test[1]; transposed = true, factor = -1)

    ## check if results are zero
    error_b1 += sum(b1.entries.^2)
    error_b2 += sum(b2.entries.^2)
    println("  eval test BLF        | error = $(error_B)")
    println("  eval test LF1        | error = $(error_b1)")
    println("  eval test LF2        | error = $(error_b2)")


    ## assemble BLF into matrix_colors
    fill!(b1.entries,0)
    fill!(b2.entries,0)
    assemble_operator!(b2[1], BLF, Test; fixed = 1, fixed_id = 1)
    assemble_operator!(b1[1], BLF, Test; fixed = 2, fixed_id = 2)
    #fill!(b2.entries,0) # once again to check if operator recast is properly reset
    #assemble_operator!(b2[1], BLF, Test; fixed = 1, fixed_id = 1)
    addblock_matmul!(b1[1], B[1,1], Test[2]; factor = -1)
    addblock_matmul!(b2[1], B[1,1], Test[1]; transposed = true, factor = -1)

    ## once again assemble full BLF to check if recast was removed successfully
    fill!(B[1,1],0)
    assemble_operator!(B[1,1], BLF)
    error_B = (lrmatmul(Test[1], B.entries, Test[2]) - 1.5)^2

    ## check if results are zero
    error_b1 += sum(b1.entries.^2)
    error_b2 += sum(b2.entries.^2)
    println("recast test BLF >> LF1 | error = $(error_b1)")
    println("recast test BLF >> LF2 | error = $(error_b2)")

    return maximum([error_B,error_b1, error_b2])
end

function test_2nd_derivs()
    ## define test function and expected operator evals
    testf = DataFunction((result,x) -> (result[1] = x[1]^2; result[2] = 3*x[2]^2 + x[1]*x[2]),[2,2]; dependencies = "X", bonus_quadorder = 2)
    expected_L = [2,6] # expected Laplacian
    expected_H = [2,0,0,0,0,1,1,6] # expected Hessian
    expected_symH = [2,0,0,0,6,1] # expected symmetric Hessian
    expected_symH2 = [2,0,0,0,6,sqrt(2)] # expected symmetric Hessian

    ## define grid = a single non-refenrece triangle
    xgrid = grid_triangle([-1.0 0.0; 1.0 0.0; 0.0 1.0]')

    ## define P2-Courant finite element space
    FEType = H1P2{2,2}
    FES = FESpace{FEType}(xgrid)

    ## get midpoint quadrature rule for constants
    qf = QuadratureRule{Float64,Triangle2D}(0)

    ## define FE basis Evaluator for Hessian
    FEBE_L = FEEvaluator(FES, Laplacian, qf)
    FEBE_H = FEEvaluator(FES, Hessian, qf)
    FEBE_symH = FEEvaluator(FES, SymmetricHessian{1}, qf)
    FEBE_symH2 = FEEvaluator(FES, SymmetricHessian{sqrt(2)}, qf)

    ## update on cell 1
    update_basis!(FEBE_L,1)
    update_basis!(FEBE_H,1)
    update_basis!(FEBE_symH,1)
    update_basis!(FEBE_symH2,1)

    ## interpolate quadratic testfunction
    Iu = FEVector(FES)
    interpolate!(Iu[1], testf)

    ## check if operator evals have the correct length
    @assert size(FEBE_L.cvals,1) == length(expected_L)
    @assert size(FEBE_H.cvals,1) == length(expected_H)
    @assert size(FEBE_symH.cvals,1) == length(expected_symH)
    @assert size(FEBE_symH2.cvals,1) == length(expected_symH2)

    ## eval operators at only quadrature point 1
    ## since function is quadratic this should be constant
    H = zeros(Float64,8)
    symH = zeros(Float64,6)
    symH2 = zeros(Float64,6)
    L = zeros(Float64,2)
    eval_febe!(L, FEBE_L, Iu.entries[FES[CellDofs][:,1]], 1)
    eval_febe!(H, FEBE_H, Iu.entries[FES[CellDofs][:,1]], 1)
    eval_febe!(symH, FEBE_symH, Iu.entries[FES[CellDofs][:,1]], 1)
    eval_febe!(symH2, FEBE_symH2, Iu.entries[FES[CellDofs][:,1]], 1)

    ## compute errors to expected values
    error_L = sqrt(sum((L - expected_L).^2))
    error_H = sqrt(sum((H - expected_H).^2))
    error_symH = sqrt(sum((symH - expected_symH).^2))
    error_symH2 = sqrt(sum((symH2 - expected_symH2).^2))
    println("EG = Triangle2D | operator = Laplacian | error = $error_L")
    println("EG = Triangle2D | operator = Hessian | error = $error_H")
    println("EG = Triangle2D | operator = SymmetricHessian{1} | error = $error_symH")
    println("EG = Triangle2D | operator = SymmetricHessian{√2} | error = $error_symH2")

    return maximum([error_L,error_H,error_symH,error_symH2])
end
