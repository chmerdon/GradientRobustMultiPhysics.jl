
using ExtendableSparse


function test_2nd_derivs()
    ## define test function and expected operator evals
    testf = DataFunction((result,x) -> (result[1] = x[1]^2; result[2] = 3*x[2]^2 + x[1]*x[2]),[2,2]; dependencies = "X", quadorder = 2)
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
    FEBE_L = FEBasisEvaluator{Float64,Triangle2D,Laplacian,ON_CELLS}(FES, qf)
    FEBE_H = FEBasisEvaluator{Float64,Triangle2D,Hessian,ON_CELLS}(FES, qf)
    FEBE_symH = FEBasisEvaluator{Float64,Triangle2D,SymmetricHessian{1},ON_CELLS}(FES, qf)
    FEBE_symH2 = FEBasisEvaluator{Float64,Triangle2D,SymmetricHessian{sqrt(2)},ON_CELLS}(FES, qf)

    ## update on cell 1
    update_febe!(FEBE_L,1)
    update_febe!(FEBE_H,1)
    update_febe!(FEBE_symH,1)
    update_febe!(FEBE_symH2,1)

    ## interpolate quadratic testfunction
    Iu = FEVector{Float64}("Iu",FES)
    interpolate!(Iu[1],testf)

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
