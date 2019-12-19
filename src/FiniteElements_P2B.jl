P2Bbasis_ref_2D = [(xref,grid,cell) -> [P2basis_ref_2D[1](xref,grid,cell),0.0],  # 1st node
                   (xref,grid,cell) -> [P2basis_ref_2D[2](xref,grid,cell),0.0],  # 2nd node
                   (xref,grid,cell) -> [P2basis_ref_2D[3](xref,grid,cell),0.0],  # 3rd node 
                   (xref,grid,cell) -> [P2basis_ref_2D[4](xref,grid,cell),0.0],       # 1st side
                   (xref,grid,cell) -> [P2basis_ref_2D[5](xref,grid,cell),0.0],  # 2nd side
                   (xref,grid,cell) -> [P2basis_ref_2D[6](xref,grid,cell),0.0],  # 3rd side
                   (xref,grid,cell) -> [27*xref[1]*xref[2]*(1 - xref[1] - xref[2]), 0.0],  # cell bubble 1st component
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[1](xref,grid,cell)], # 1st node
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[2](xref,grid,cell)], # 2nd node
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[3](xref,grid,cell)], # 3rd node
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[4](xref,grid,cell)], # 1st side
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[5](xref,grid,cell)], # 2nd side
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[6](xref,grid,cell)],
                   (xref,grid,cell) -> [0.0, 27*xref[1]*xref[2]*(1 - xref[1] - xref[2])]]  # cell bubble 2nd component


function get_P2BFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    nfaces::Int = size(grid.nodes4faces,1);
    
    # group basis functions
    xdim = size(grid.coords4nodes,2);
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        dofs4cells = zeros(Int64,ncells,14);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4:6] = nnodes .+ grid.faces4cells;
        dofs4cells[:,7] = nnodes + nfaces .+ Array(1:ncells);
        dofs4cells[:,8:10] = (nnodes + nfaces + ncells) .+ grid.nodes4cells;
        dofs4cells[:,11:13] = (2*nnodes + nfaces + ncells) .+ grid.faces4cells;
        dofs4cells[:,14] = 2*(nnodes + nfaces) +ncells .+ Array(1:ncells);
        dofs4faces = zeros(Int64,nfaces,6);
        dofs4faces[:,[1,3]] = grid.nodes4faces;
        dofs4faces[:,2] = nnodes .+ Array(1:nfaces);
        dofs4faces[:,[4,6]] = (nnodes + nfaces + ncells) .+ grid.nodes4faces;
        dofs4faces[:,5] = (2*nnodes + nfaces + ncells) .+ Array(1:nfaces);
        xref4dofs4cell = [0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5; 1//3 1//3; 0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5; 1//3 1//3];
        
        bfun_ref = P2Bbasis_ref_2D;
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D Vector P2B-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:],2);
            end
        else                  
            println("Initialising 2D Vector P2B-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_P2_1_grad!(0),
                          triangle_P2_2_grad!(0),
                          triangle_P2_3_grad!(0),
                          triangle_P2_4_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_5_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_6_grad!(grid.coords4nodes[1,:],0),
                          triangle_cellbubble_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_1_grad!(2),
                          triangle_P2_2_grad!(2),
                          triangle_P2_3_grad!(2),
                          triangle_P2_4_grad!(grid.coords4nodes[1,:],2),
                          triangle_P2_5_grad!(grid.coords4nodes[1,:],2),
                          triangle_P2_6_grad!(grid.coords4nodes[1,:],2),
                          triangle_cellbubble_grad!(grid.coords4nodes[1,:],2)];
                      
        end
    end    
    
    return FiniteElement{T}("P2B=(P2V+CBV)", grid, 3, celldim - 1, (celldim - 1)*(nnodes+nfaces+ncells), dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

# all exact gradients can be found in FiniteElement_Lagrange
