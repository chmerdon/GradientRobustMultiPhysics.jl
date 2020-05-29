using ExtendableSparse
using BenchmarkTools

function main(ncells::Int64 = 100, ndofs4cell::Int64 = 12)
    ndofs::Int64 = ncells*ndofs4cell
    dofs4cell = 1:ndofs4cell
    nqp = 5
    
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)

    val = 0.0
    @time begin
        for cell = 1:ncells
            dofs4cell = rand(1:ndofs,12);
            for q = 1:nqp
                for j=1:ndofs4cell, k=1:ndofs4cell
                    val = rand()
                    A[dofs4cell[j],dofs4cell[k]] += val
                end
                for j =1:ndofs4cell
                    A[ndofs,dofs4cell[j]] += val
                    A[dofs4cell[j],ndofs] += val
                end    
            end    
        end
    end    
    return A
end

main(100,12)
main(1000,12)
main(2000,12)