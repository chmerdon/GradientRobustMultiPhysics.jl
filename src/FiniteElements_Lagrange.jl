
P1basis_ref_1D = [(xref,grid,cell) -> 1 - xref[1],  # 1st node
                  (xref,grid,cell) -> xref[1]]; # 2nd node
                  
P1basis_ref_2D = [(xref,grid,cell) -> 1 - xref[1] - xref[2],  # 1st node
                  (xref,grid,cell) -> xref[1],  # 2nd node
                  (xref,grid,cell) -> xref[2]]; # 3rd node
                  
P1basis_ref_3D = [(xref,grid,cell) -> 1 - xref[1] - xref[2] - xref[3],  # 1st node
                  (xref,grid,cell) -> xref[1],  # 2nd node
                  (xref,grid,cell) -> xref[2],  # 3rd node
                  (xref,grid,cell) -> xref[3]]; # 4th node
               
P2basis_ref_1D = [(xref,grid,cell) -> 2*(1 - xref[1])*(1//2 - xref[1]), # 1st node
                  (xref,grid,cell) -> 2*xref[1]*(xref[1] - 1//2), # 2nd node
                  (xref,grid,cell) -> 4*(1 - xref[1])*xref[1]]  # midpoint
               
               
P2basis_ref_2D = [(xref,grid,cell) -> 2*(1 - xref[1] - xref[2])*(1//2 - xref[1] - xref[2]), # 1st node
                  (xref,grid,cell) -> 2*xref[1]*(xref[1] - 1//2), # 2nd node
                  (xref,grid,cell) -> 2*xref[2]*(xref[2] - 1//2), # 3rd node 
                  (xref,grid,cell) -> 4*(1 - xref[1] - xref[2])*xref[1],  # 1st side
                  (xref,grid,cell) -> 4*xref[1]*xref[2],  # 2nd side
                  (xref,grid,cell) -> 4*xref[2]*(1 - xref[1] - xref[2])]; # 3rd side
               
P1Vbasis_ref_2D = [(xref,grid,cell) -> [1 - xref[1] - xref[2], 0.0],  # 1st node 1st component
                   (xref,grid,cell) -> [xref[1], 0.0],  # 2nd node 1st component
                   (xref,grid,cell) -> [xref[2], 0.0],  # 3rd node 1st component
                   (xref,grid,cell) -> [0.0, 1 - xref[1] - xref[2]],  # 1st node 2nd component
                   (xref,grid,cell) -> [0.0, xref[1]],  # 2nd node 2nd component
                   (xref,grid,cell) -> [0.0, xref[2]]]; # 3rd node 2nd component 

P2Vbasis_ref_2D = [(xref,grid,cell) -> [P2basis_ref_2D[1](xref,grid,cell),0.0],  # 1st node
                   (xref,grid,cell) -> [P2basis_ref_2D[2](xref,grid,cell),0.0],  # 2nd node
                   (xref,grid,cell) -> [P2basis_ref_2D[3](xref,grid,cell),0.0],  # 3rd node 
                   (xref,grid,cell) -> [P2basis_ref_2D[4](xref,grid,cell),0.0],       # 1st side
                   (xref,grid,cell) -> [P2basis_ref_2D[5](xref,grid,cell),0.0],  # 2nd side
                   (xref,grid,cell) -> [P2basis_ref_2D[6](xref,grid,cell),0.0],  # 3rd side
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[1](xref,grid,cell)], # 1st node
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[2](xref,grid,cell)], # 2nd node
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[3](xref,grid,cell)], # 3rd node
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[4](xref,grid,cell)], # 1st side
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[5](xref,grid,cell)], # 2nd side
                   (xref,grid,cell) -> [0.0, P2basis_ref_2D[6](xref,grid,cell)]] # 3rd side

 #########################
 ### P0 FINITE ELEMENT ###
 #########################

function get_P0FiniteElement(grid::Grid.Mesh)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    dofs4cells = zeros(Int64,ncells,1);
    dofs4cells[:,1] = 1:ncells;
    dofs4faces = [[] []];
    
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    xref4dofs4cell = Matrix{Float64}(undef,1,celldim-1);
    fill!(xref4dofs4cell,1/(celldim-1)) 
    
    # set basis functions and gradients
    if celldim == 3
        trafo = local2global_triangle();
    elseif celldim == 4
        trafo = local2global_line();
    end    
    bfun_ref = Vector{Function}(undef,1)
    bfun_grad! = Vector{Function}(undef,1)
    bfun_ref[1] = (xref,grid,cell) -> 1;
    bfun_grad![1] = function P0gradient(result,xref,grid,cell) 
                      result[:] .= 0
                    end  
    
    return FiniteElement{T}("P0", grid, 0, 1, ncells, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end 


 #################################################
 ### COURANT P1 FINITE ELEMENT (H1-conforming) ###
 #################################################
  
function get_P1FiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    dofs4cells = grid.nodes4cells;
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    dofs4faces = grid.nodes4faces;
    nnodes = size(grid.coords4nodes,1);
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    if celldim == 3 # triangles
        bfun_ref = P1basis_ref_2D;
        xref4dofs4cell = [0 0; 1 0; 0 1];
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo,bfun_ref[k],grid.coords4nodes[1,:])
            end
        else
            println("Initialising 2D P1-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0)];
        end
    elseif celldim == 2 # line segments
        bfun_ref = P1basis_ref_1D;
        xref4dofs4cell = zeros(Float64,2,1)
        xref4dofs4cell[1,:] = [0];
        xref4dofs4cell[2,:] = [1];
        trafo = local2global_line();
        if FDgradients
            println("Initialising 1D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo,bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else
            println("Initialising 1D P1-FiniteElement with exact gradients...");
            bfun_grad! = [line_bary1_grad!,
                          line_bary2_grad!];
        end
    end    
    
    return FiniteElement{T}("P1", grid, 1, 1, nnodes, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


function get_P1VectorFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    nnodes::Int = size(grid.coords4nodes,1);
    ncells::Int = size(grid.nodes4cells,1);
    dofs4cells = zeros(Int64,ncells,(celldim-1)*celldim);
    dofs4cells[:,1:celldim] = grid.nodes4cells
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    nfaces = size(grid.nodes4faces,1);
    dofs4faces = zeros(Int64,nfaces,(celldim-1)*(celldim-1));
    dofs4faces[:,1:celldim-1] = grid.nodes4faces
    
    # group basis functions
    if celldim == 3 # triangles
        dofs4cells[:,celldim+1:2*celldim] = nnodes.+grid.nodes4cells;
        dofs4faces[:,celldim:2*(celldim-1)] = nnodes.+grid.nodes4faces;
        xref4dofs4cell = [0 0; 1 0; 0 1; 0 0; 1 0; 0 1];
        bfun_ref = P1Vbasis_ref_2D;
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D Vector P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo,bfun_ref[k],grid.coords4nodes[1,:],2);
            end
        else
            println("Initialising 2D Vector P1-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0),
                          triangle_bary1_grad!(2),
                          triangle_bary2_grad!(2),
                          triangle_bary3_grad!(2)];
        end
    elseif celldim == 2 # line segments
        xref4dofs4cell = zeros(Float64,4,1)
        xref4dofs4cell[1,:] = [0];
        xref4dofs4cell[2,:] = [1];
        xref4dofs4cell[3,:] = [0];
        xref4dofs4cell[4,:] = [1];
        bfun_ref = P1basis_ref_1D;
        trafo = local2global_line();
        if FDgradients
            println("Initialising 1D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo,bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else
            println("Initialising 1D P1-FiniteElement with exact gradients...");
            bfun_grad! = [line_bary1_grad!,
                          line_bary2_grad!];
        end
    end  
    
    return FiniteElement{T}("P1", grid, 1, celldim-1, (celldim-1)*nnodes, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


 ##################################################
 ### LAGRANGE P2 FINITE ELEMENT (H1-conforming) ###
 ##################################################

function get_P2FiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
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
        dofs4cells = [grid.nodes4cells (nnodes .+ grid.faces4cells)];
        dofs4faces = [grid.nodes4faces[:,1] 1:size(grid.nodes4faces,1) grid.nodes4faces[:,2]];
        dofs4faces[:,2] .+= nnodes;
        xref4dofs4cell = [0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5];
        
        bfun_ref = P2basis_ref_2D;
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else                  
            println("Initialising 2D P2-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_P2_1_grad!(0),
                          triangle_P2_2_grad!(0),
                          triangle_P2_3_grad!(0),
                          triangle_P2_4_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_5_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_6_grad!(grid.coords4nodes[1,:],0)];
                      
        end   
    elseif celldim == 2 # line segments
        dofs4cells = [grid.nodes4cells 1:ncells];
        dofs4cells[:,3] .+= nnodes;
        dofs4faces = grid.nodes4faces;
        xref4dofs4cell = zeros(Float64,3,1)
        xref4dofs4cell[1,:] = [0];
        xref4dofs4cell[2,:] = [1];
        xref4dofs4cell[3,:] = [0.5];
        
        bfun_ref = P2basis_ref_1D;
        trafo = local2global_line();
        if FDgradients
            println("Initialising 1D P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else
            println("Initialising 1D P2-FiniteElement with exact gradients...");
            bfun_grad! = [line_P2_1_grad!,
                          line_P2_2_grad!,
                          line_P2_3_grad!(grid.coords4nodes[1,:])];
        end
    end    
    
    return FiniteElement{T}("P2", grid, 2, 1, nnodes + nfaces, dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


function get_P2VectorFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
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
        dofs4cells = zeros(Int64,ncells,12);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4:6] = nnodes .+ grid.faces4cells;
        dofs4cells[:,7:9] = (nnodes + nfaces) .+ grid.nodes4cells;
        dofs4cells[:,10:12] = (2*nnodes + nfaces) .+ grid.faces4cells;
        dofs4faces = zeros(Int64,nfaces,6);
        dofs4faces[:,[1,3]] = grid.nodes4faces;
        dofs4faces[:,2] = nnodes .+ Array(1:nfaces);
        dofs4faces[:,[4,6]] = (nnodes + nfaces) .+ grid.nodes4faces;
        dofs4faces[:,5] = (2*nnodes + nfaces) .+ Array(1:nfaces);
        xref4dofs4cell = [0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5; 0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5];
        
        bfun_ref = P2Vbasis_ref_2D;
        trafo = local2global_triangle();
        if FDgradients
            println("Initialising 2D Vector P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:],2);
            end
        else                  
            println("Initialising 2D Vector P2-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_P2_1_grad!(0),
                          triangle_P2_2_grad!(0),
                          triangle_P2_3_grad!(0),
                          triangle_P2_4_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_5_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_6_grad!(grid.coords4nodes[1,:],0),
                          triangle_P2_1_grad!(2),
                          triangle_P2_2_grad!(2),
                          triangle_P2_3_grad!(2),
                          triangle_P2_4_grad!(grid.coords4nodes[1,:],2),
                          triangle_P2_5_grad!(grid.coords4nodes[1,:],2),
                          triangle_P2_6_grad!(grid.coords4nodes[1,:],2)];
                      
        end   
    elseif celldim == 2 # line segments
        dofs4cells = [grid.nodes4cells 1:ncells];
        dofs4cells[:,3] .+= nnodes;
        dofs4faces = grid.nodes4faces;
        xref4dofs4cell = zeros(Float64,3,1)
        xref4dofs4cell[1,:] = [0];
        xref4dofs4cell[2,:] = [1];
        xref4dofs4cell[3,:] = [0.5];
        
        bfun_ref = P2basis_ref_1D;
        trafo = local2global_line();
        if FDgradients
            println("Initialising 1D P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun_ref));
            for k = 1:length(bfun_ref)
                bfun_grad![k] = FDgradient2(trafo, bfun_ref[k],grid.coords4nodes[1,:]);
            end
        else
            println("Initialising 1D P2-FiniteElement with exact gradients...");
            bfun_grad! = [line_P2_1_grad!,
                          line_P2_2_grad!,
                          line_P2_3_grad!(grid.coords4nodes[1,:])];
        end
    end    
    
    return FiniteElement{T}("P2", grid, 2, celldim - 1, (celldim-1)*(nnodes + nfaces), dofs4cells, dofs4faces, xref4dofs4cell, trafo, bfun_ref, bfun_grad!);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

# the two exact gradients of the P1 basis functions on a line
function line_bary1_grad!(result,xref,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = -1 / grid.volume4cells[cell];
end   
function line_bary2_grad!(result,xref,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = 1 / grid.volume4cells[cell];
end   


# the three exact gradients of the P2 basis functions on a line
function line_P2_1_grad!(result,xref,grid,cell)
    line_bary1_grad!(result,xref,grid,cell);
    result .*= (3 - 4*xref[1]);
end   
function line_P2_2_grad!(result,xref,grid,cell)
    line_bary2_grad!(result,xref,grid,cell);
    result .*= (4*xref[1]-1);
end   
function line_P2_3_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        line_bary1_grad!(temp,xref,grid,cell)
        line_bary2_grad!(result,xref,grid,cell)
        for j = 1 : length(x)
            result[j] = 4*(temp[j] .* xref[1] + result[j] .* (1 - xref[1]));
        end
    end
end


# the three exact gradients of the P1 basis functions on a triangle
function triangle_bary1_grad!(offset)
    function closure(result,xref,grid,cell)
        result[1+offset] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,3],2];
        result[2+offset] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,2],1];
        result ./= (2*grid.volume4cells[cell]);
    end    
end
function triangle_bary2_grad!(offset)
    function closure(result,xref,grid,cell)
        result[1+offset] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2];
        result[2+offset] = grid.coords4nodes[grid.nodes4cells[cell,1],1] - grid.coords4nodes[grid.nodes4cells[cell,3],1];
        result ./= (2*grid.volume4cells[cell]);
    end    
end
function triangle_bary3_grad!(offset)
    function closure(result,xref,grid,cell)
        result[1+offset] = grid.coords4nodes[grid.nodes4cells[cell,1],2] - grid.coords4nodes[grid.nodes4cells[cell,2],2];
        result[2+offset] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1];
        result ./= (2*grid.volume4cells[cell]);
    end
end


# the six exact gradients of the P2 basis functions on a triangle
function triangle_P2_1_grad!(offset)
    function closure(result,xref,grid,cell)
        triangle_bary1_grad!(offset)(result,xref,grid,cell);
        result[offset+1] *= (3 - 4*(xref[1]+xref[2]));
        result[offset+2] *= (3 - 4*(xref[1]+xref[2]));
    end
end
function triangle_P2_2_grad!(offset)
    function closure(result,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,xref,grid,cell)
        result[offset+1] *= (4*xref[1]-1);
        result[offset+2] *= (4*xref[1]-1);
    end    
end
function triangle_P2_3_grad!(offset)
    function closure(result,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,xref,grid,cell)
        result[offset+1] *= (4*xref[2]-1);
        result[offset+2] *= (4*xref[2]-1);
    end    
end
function triangle_P2_4_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* xref[1] + result[offset+j] .* (1 - xref[1] - xref[2]));
        end
    end    
end
function triangle_P2_5_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary2_grad!(0)(temp,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* xref[2] + result[offset+j] .* xref[1]);
        end
    end
end
function triangle_P2_6_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary3_grad!(0)(temp,xref,grid,cell)
        triangle_bary1_grad!(offset)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* (1 - xref[1] - xref[2]) + result[offset+j] .* xref[2]);
        end
    end
end


function triangle_cellbubble_grad!(x, offset)
    temp = zeros(eltype(x),length(x));
    temp2 = zeros(eltype(x),length(x));
    function closure(result,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,xref,grid,cell)
        triangle_bary2_grad!(0)(temp2,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 27*(temp[j]*xref[1]*xref[2] + temp2[j]*(1-xref[1]-xref[2])*xref[2] + result[offset+j]*(1-xref[1]-xref[2])*xref[1])
        end
    end    
end
