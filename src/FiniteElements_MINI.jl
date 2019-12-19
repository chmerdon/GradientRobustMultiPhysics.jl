MINIbasis_ref_2D = [(xref,grid,cell) -> [1 - xref[1] - xref[2], 0.0],  # 1st node 1st component
                    (xref,grid,cell) -> [xref[1], 0.0],  # 2nd node 1st component
                    (xref,grid,cell) -> [xref[2], 0.0],  # 3rd node 1st component
                    (xref,grid,cell) -> [27*xref[1]*xref[2]*(1 - xref[1] - xref[2]), 0.0],  # cell bubble 1st component
                    (xref,grid,cell) -> [0.0, 1 - xref[1] - xref[2]],  # 1st node 2nd component
                    (xref,grid,cell) -> [0.0, xref[1]],  # 2nd node 2nd component
                    (xref,grid,cell) -> [0.0, xref[2]],  # 3rd node 2nd component 
                    (xref,grid,cell) -> [0.0, 27*xref[1]*xref[2]*(1 - xref[1] - xref[2])]]  # cell bubble 2nd component


function get_MINIFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
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
        dofs4cells = zeros(Int64,ncells,8);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4] = nnodes .+ Array(1:ncells);
        dofs4cells[:,5:7] = (ncells + nnodes) .+ grid.nodes4cells;
        dofs4cells[:,8] = (ncells + 2*nnodes) .+ Array(1:ncells);
        dofs4faces = zeros(Int64,nfaces,4);
        dofs4faces[:,[1,2]] = grid.nodes4faces;
        dofs4faces[:,[3,4]] = (ncells + nnodes) .+ grid.nodes4faces;
        xref4dofs4cell = [0 0; 1 0; 0 1; 1//3 1//3;0 0; 1 0; 0 1; 1//3 1//3];
        trafo = local2global_triangle();
        
        bfun_ref = MINIbasis_ref_2D;
        if FDgradients
            println("Initialising 2D MINI-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:],xdim);
            end
        else                  
            println("Initialising 2D MINI-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0),
                          triangle_cellbubble_grad!(grid.coords4nodes[1,:],0),
                          triangle_bary1_grad!(2),
                          triangle_bary2_grad!(2),
                          triangle_bary3_grad!(2),
                          triangle_cellbubble_grad!(grid.coords4nodes[1,:],2)];
                      
        end   
    end    
    
    return FiniteElement{T}("MINI=(P1V+CBV)", grid, 3, celldim - 1, (celldim - 1)*(nnodes+ncells), dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

# all exact gradients can be found in FiniteElement_Lagrange
