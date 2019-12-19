module FiniteElementsTests

using Grid
using FiniteElements
using LinearAlgebra
using DiffResults
using Quadrature

export FiniteElement


function TestFEConsistency(FE::FiniteElements.FiniteElement, cellnr, check_gradients::Bool = true)
    ndof4cell = size(FE.dofs4cells,2);
    celldim = size(FE.grid.nodes4cells,2);
    xdim = size(FE.grid.coords4nodes,2);
    T = eltype(FE.grid.coords4nodes);
    basiseval_ref = zeros(T,ndof4cell,ndof4cell);
    allok = true;
    gradient_exact = zeros(T,ndof4cell,ndof4cell,xdim*FE.ncomponents);
    gradient_FD = zeros(T,ndof4cell,ndof4cell,xdim*FE.ncomponents);
    x = zeros(T,xdim);
    xref = zeros(T,xdim+1);
    if check_gradients
        FDgradients = Vector{Function}(undef,ndof4cell);
        for j = 1 : ndof4cell
            FDgradients[j] = FiniteElements.FDgradient2(FE.loc2glob_trafo,FE.bfun_ref[j],x,FE.ncomponents)
        end    
    end
    for j = 1 : ndof4cell
        xref = FE.xref4dofs4cell[j,:];
        x = FE.loc2glob_trafo(FE.grid,cellnr)(xref);
        println("\ncoordinate of dof nr ",j);
        print("x = "); show(x); println("");
        print("xref = "); show(xref); println("");
        for k = 1 : ndof4cell
            basiseval_ref[j,k] = maximum(FE.bfun_ref[k](xref,FE.grid,cellnr))
            FE.bfun_grad![k](view(gradient_exact,j,k,:),xref,FE.grid,cellnr);
            if check_gradients
                FDgradients[k](view(gradient_FD,j,k,:),xref,FE.grid,cellnr);
            end    
        end
        println("\neval of basis functions in xref at dof nr ",j);
        show(basiseval_ref[j,:]);
        println("\neval of active gradients of basis functions at dof nr ",j);
        for k = 1 : ndof4cell
            show(gradient_exact[j,k,:]); println("");
        end    
        if check_gradients
            println("eval of ForwardDiff gradients of basis functions at dof nr ",j);
            for k = 1 : ndof4cell
                show(gradient_FD[j,k,:]); println("");
            end
        end
    end

    if check_gradients
        println("\nVERDICT:");
        if norm(gradient_exact - gradient_FD) > eps(1.0)
            allok = false
            println("gradients of basis functions seem wrong");
        else
            println("gradients of basis functions seem ok");
        end
    end    
    return allok;
end


function TestP0()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P0FiniteElement(grid);
    TestFEConsistency(FE,1,true);
end

function TestP1_1D()
    # generate reference domain
    coords4nodes_init = zeros(Rational,2,1);
    coords4nodes_init[:] = [0 1];
    nodes4cells_init = zeros(Int64,1,2);
    nodes4cells_init[1,:] = [1 2];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    TestFEConsistency(FE,1,false);
end

function TestP2_1D()
    # generate reference domain
    coords4nodes_init = zeros(Rational,2,1);
    coords4nodes_init[:] = [0 1];
    nodes4cells_init = zeros(Int64,1,2);
    nodes4cells_init[1,:] = [1 2];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2FiniteElement(grid,false);
    TestFEConsistency(FE,1,true);
end

function TestP1()
    # generate reference domain
    coords4nodes_init = [0 1//2;
                        2 0;
                        0 1];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


function TestP1V()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1VectorFiniteElement(grid,false);
    TestFEConsistency(FE,1,true);
end


function TestBR()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_BRFiniteElement(grid,false);
    TestFEConsistency(FE,1,true);
end

function TestP2()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2FiniteElement(grid,false);
    @time TestFEConsistency(FE,1,true);
end

function TestP2V()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2VectorFiniteElement(grid,false);
    @time TestFEConsistency(FE,1,true);
end

function TestCR()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_CRFiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


function TestCR_3D()
    # generate reference domain
    coords4nodes_init = [0.0 0.0 0.0;
                        1.0 0.0 0.0;
                        0.0 1.0 0.0;
                        0.0 0.0 1.0];
    nodes4cells_init = zeros(Int64,1,4);
    nodes4cells_init[1,:] = [1 2 3 4];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_CRFiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


end # module
