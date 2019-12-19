BRbasis_ref_2D = [(xref,grid,cell) -> [1 - xref[1] - xref[2], 0.0],  # 1st node 1st component
                  (xref,grid,cell) -> [xref[1], 0.0],  # 2nd node 1st component
                  (xref,grid,cell) -> [xref[2], 0.0],  # 3rd node 1st component
                  (xref,grid,cell) -> [0.0, 1 - xref[1] - xref[2]],  # 1st node 2nd component
                  (xref,grid,cell) -> [0.0, xref[1]],  # 2nd node 2nd component
                  (xref,grid,cell) -> [0.0, xref[2]],  # 3rd node 2nd component 
                  (xref,grid,cell) -> 4*(1 - xref[1] - xref[2])*xref[1] .* grid.normal4faces[grid.faces4cells[cell,1],:],  # 1st side
                  (xref,grid,cell) -> 4*xref[1]*xref[2] .* grid.normal4faces[grid.faces4cells[cell,2],:],  # 2nd side
                  (xref,grid,cell) -> 4*xref[2]*(1 - xref[1] - xref[2]) .* grid.normal4faces[grid.faces4cells[cell,3],:]]; # 3rd side
               

function get_BRFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_normal4faces!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    nfaces::Int = size(grid.nodes4faces,1);
    
    # group basis functions
    xdim = size(grid.coords4nodes,2);
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        dofs4cells = zeros(Int64,ncells,9);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4:6] = nnodes .+ grid.nodes4cells;
        dofs4cells[:,7:9] = 2*nnodes .+ grid.faces4cells;
        dofs4faces = zeros(Int64,nfaces,5);
        dofs4faces[:,[1,3]] = grid.nodes4faces;
        dofs4faces[:,[2,4]] = nnodes .+ grid.nodes4faces;
        dofs4faces[:,5] = 2*nnodes .+ Array(1:nfaces);
        
        bfun_ref = BRbasis_ref_2D;
        xref4dofs4cell = [0.0 0.0; 1.0 0.0; 0.0 0.1;0.0 0.0; 1.0 0.0; 0.0 0.1; 0.5 0.0; 0.0 0.5; 0.5 0.5];
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D Bernardi-Raugel-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:], 2);
            end
        else                  
            println("Initialising 2D Bernardi-Raugel-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0),
                          triangle_bary1_grad!(2),
                          triangle_bary2_grad!(2),
                          triangle_bary3_grad!(2),
                          triangle_BR_1_grad!(grid.coords4nodes[1,:]),
                          triangle_BR_2_grad!(grid.coords4nodes[1,:]),
                          triangle_BR_3_grad!(grid.coords4nodes[1,:])];
                      
        end   
    end    
    
    return FiniteElement{T}("BR=(P1V+nFB)", grid, 2, celldim - 1, (celldim-1)*nnodes + nfaces, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

function triangle_BR_1_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,xref,grid,cell)
        triangle_bary2_grad!(0)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[2+j] = result[j]
            result[j] = 4*(temp[j] .* xref[1] + result[j] .* (1 - xref[1] - xref[2])) * grid.normal4faces[grid.faces4cells[cell,1],1];
            result[2+j] = 4*(temp[j] .* xref[1] + result[2+j] .* (1 - xref[1] - xref[2])) * grid.normal4faces[grid.faces4cells[cell,1],2];
        end
    end    
end


function triangle_BR_2_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary2_grad!(0)(temp,xref,grid,cell)
        triangle_bary3_grad!(0)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[2+j] = result[j]
            result[j] = 4*(temp[j] .* xref[2] + result[j] .* xref[1]) * grid.normal4faces[grid.faces4cells[cell,2],1];
            result[2+j] = 4*(temp[j] .* xref[2] + result[2+j] .* xref[1]) * grid.normal4faces[grid.faces4cells[cell,2],2];
        end
    end    
end

function triangle_BR_3_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary3_grad!(0)(temp,xref,grid,cell)
        triangle_bary1_grad!(0)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[2+j] = result[j]
            result[j] = 4*(temp[j] .* (1 - xref[1] - xref[2]) + result[j] .* xref[2]) * grid.normal4faces[grid.faces4cells[cell,3],1];
            result[2+j] = 4*(temp[j] .* (1 - xref[1] - xref[2]) + result[2+j] .* xref[2]) * grid.normal4faces[grid.faces4cells[cell,3],2];
        end
    end    
end
