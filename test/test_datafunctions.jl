
function run_datafunction_tests()
    @testset "DataFunctions" begin
        println("\n")
        println("=====================")
        println("Testing DataFunctions")
        println("=====================")
        
        ## test 1st and 2nd order space derivatives
        function eval_u!(result,x)
            result[1] = x[1]^3+x[3]^2
            result[2] = -x[1]^2 + x[2] + 1
            result[3] = x[1]*x[2]
        end
        u = DataFunction(eval_u!, [3,3]; name = "u", dependencies = "X", bonus_quadorder = 3)

        real_divu(x) = [3*x[1]^2 + 1]
        real_Δu(x) = [6*x[1] + 2, -2, 0]
        real_∇u(x) = [3*x[1]^2 0 2*x[3]
                      -2*x[1] 1 0
                      x[2] x[1] 0]
        real_Hu(x) = [6*x[1] 0 0    ##=d(du1/dx1)
                        -2 0 0      ##=d(du2/dx1)
                        0 1 0       ##=d(du3/dx1)
                      0 0 0         ##=d(du1/dx2)
                      0 0 0         ##=d(du2/dx2)
                      1 0 0         ##=d(du3/dx2)
                      0 0 2         ##=d(du1/dx3)
                      0 0 0         ##=d(du2/dx3)
                      0 0 0]        ##=d(du3/dx3)

        ## check AD derivatives
        divu = eval_div(u)
        ∇u = eval_∇(u)
        Hu = eval_H(u)
        Δu = eval_Δ(u)

        xtest = [1.3, 0.7, -0.1]
        @test divu(xtest) == real_divu(xtest)
        @test ∇u(xtest) == real_∇u(xtest)
        @test Hu(xtest) == real_Hu(xtest)
        @test Δu(xtest) == real_Δu(xtest)

        # Hessian comparison currently a bit ugly
        # difficulty:
        # H(u) is a DataFunction that returns vectors with all entries of Hessian
        # eval_H(u) is a Function that returns matrices with entries of Hessian (matching real_Hu)
        # 
        # it holds:
        # H(u) == Hreshape(eval_H(u))
        #
        function Hreshape(H)
            result = zeros(Float64,27)
            for u = 1 : 3
                for xj = 1 : 3
                    for xk = 1 : 3
                        result[(u-1)*9 + (xj-1)*3 + xk] = H[(xj-1)*3 + u, xk]
                    end
                end
            end
            return result
        end

        ## check DataFunction-ized AD derivatives
        divu = div(u)
        ∇u = ∇(u)
        Hu = H(u)
        Δu = Δ(u)
        @test divu(xtest) == view(real_divu(xtest),:)
        @test ∇u(xtest) == view(transpose(real_∇u(xtest)),:)
        @test Hu(xtest) == Hreshape(real_Hu(xtest))
        @test Δu(xtest) == view(real_Δu(xtest),:)


        ## test space and time derivative for time-dependent field
        function eval_u2!(result,x,t)
            result[1] = t^3*(x[1]^3+x[3]^2)
            result[2] = t^2*(-x[1]^2 + x[2] + 1)
            result[3] = t*(x[1]*x[2])
        end
        u = DataFunction(eval_u2!, [3,3]; name = "u", dependencies = "XT", bonus_quadorder = 3)
        
        real_dtu(x,t) = [3*t^2*(x[1]^3+x[3]^2), 2*t*(-x[1]^2 + x[2] + 1), (x[1]*x[2])]
        real_divu(x,t) = [t^3*3*x[1]^2 + t^2]
        real_Δu(x,t) = [t^3*(6*x[1] + 2), -2*t^2, 0]

        divu = eval_div(u)
        ∇u = eval_∇(u)
        Hu = eval_H(u)
        Δu = eval_Δ(u)
        dtu = eval_dt(u)

        xtest, ttest = [0.3, 0.4, -0.9], 1.7
        @test divu(xtest,ttest) == real_divu(xtest,ttest)
        @test Δu(xtest,ttest) == real_Δu(xtest,ttest)
        @test dtu(xtest,ttest) == real_dtu(xtest,ttest)

    end
end

