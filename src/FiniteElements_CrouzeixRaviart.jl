CRbasis_ref_2D = [(xref,grid,cell) -> 1 - 2*xref[2],  # 1st side (opposite to node 3)
                  (xref,grid,cell) -> 2*(xref[1]+xref[2]) - 1,  # 2nd side (opposite to node 1)
                  (xref,grid,cell) -> 1 - 2*xref[1]]; # 3rd side (opposite to node 2)
    
CRVbasis_ref_2D = [(xref,grid,cell) -> [1 - 2*xref[2], 0.0],  # 1st side (opposite to node 3)
                   (xref,grid,cell) -> [2*(xref[1]+xref[2]) - 1, 0.0],  # 2nd side (opposite to node 1)
                   (xref,grid,cell) -> [1 - 2*xref[1], 0.0],
                   (xref,grid,cell) -> [0.0, 1 - 2*xref[2]],  # 1st side (opposite to node 3)
                   (xref,grid,cell) -> [0.0, 2*(xref[1]+xref[2]) - 1],  # 2nd side (opposite to node 1)
                   (xref,grid,cell) -> [0.0, 1 - 2*xref[1]]]; # 3rd side (opposite to node 2)
    
    
function get_CRFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    ensure_faces4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nfaces::Int = size(grid.nodes4faces,1);
    dofs4cells = grid.faces4cells;
    dofs4faces = zeros(Int64,nfaces,1);
    dofs4faces[:,1] = 1:nfaces;
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    @assert celldim >= 3
    if celldim == 3 # triangles
        bfun_ref = CRbasis_ref_2D;
        xref4dofs4cell = [0.5 0.0; 0.0 0.5; 0.5 0.5];
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else
            println("Initialising 2D CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR_1_grad!(0),
                          triangle_CR_2_grad!(0),
                          triangle_CR_3_grad!(0)];
        end
    end    
    
    return FiniteElement{T}("CR", grid,1, 1, nfaces, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end

function get_CRVFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    ensure_faces4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nfaces::Int = size(grid.nodes4faces,1);
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    @assert celldim >= 3
    if celldim == 3 # triangles
        dofs4cells = zeros(Int64,ncells,6);
        dofs4cells[:,1:3] = grid.faces4cells;
        dofs4cells[:,4:6] = nfaces .+ grid.faces4cells;
        dofs4faces = zeros(Int64,nfaces,2);
        dofs4faces[:,1] = 1:nfaces;
        dofs4faces[:,2] = nfaces .+ Array(1:nfaces);
        
        bfun_ref = CRVbasis_ref_2D;
        xref4dofs4cell = [0.5 0.0; 0.0 0.5; 0.5 0.5; 0.5 0.0; 0.0 0.5; 0.5 0.5];
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D Vector CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else
            println("Initialising 2D Vector CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR_1_grad!(0),
                          triangle_CR_2_grad!(0),
                          triangle_CR_3_grad!(0),
                          triangle_CR_1_grad!(2),
                          triangle_CR_2_grad!(2),
                          triangle_CR_3_grad!(2)];
        end
    end    
    
    return FiniteElement{T}("CR", grid, 1, celldim-1, (celldim-1)*nfaces, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

function triangle_CR_1_grad!(offset)
    function closure(result,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,xref,grid,cell);
        result[1+offset] *= -2;
        result[2+offset] *= -2;
    end    
end
function triangle_CR_2_grad!(offset)
    function closure(result,xref,grid,cell)
        triangle_bary1_grad!(offset)(result,xref,grid,cell);
        result[1+offset] *= -2;
        result[2+offset] *= -2;
    end    
end
function triangle_CR_3_grad!(offset)
    function closure(result,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,xref,grid,cell);
        result[1+offset] *= -2;
        result[2+offset] *= -2;
    end    
end
