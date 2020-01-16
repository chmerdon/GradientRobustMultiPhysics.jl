struct H1P2FiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getP2FiniteElement(grid,ncomponents)
    ensure_faces4cells!(grid);
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    if dim == 1
        xref4dofs4cell = repeat(Array{Float64,2}([0,1,0.5]')',ncomponents);
    elseif dim == 2    
        xref4dofs4cell = repeat([0 0; 1 0; 0 1; 0.5 0; 0.5 0.5; 0 0.5],ncomponents);
    end 
    return H1P2FiniteElement{T,dim,ncomponents}("P2 (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end   

get_polynomial_order(FE::H1P2FiniteElement) = 2;

get_ndofs(FE::H1P2FiniteElement{T,1,1} where T <: Real) = size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4cells,1);
get_ndofs(FE::H1P2FiniteElement{T,2,1} where T <: Real) = size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1);
get_ndofs(FE::H1P2FiniteElement{T,2,2} where T <: Real) = 2*(size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1));

get_xref4dof(FE::H1P2FiniteElement, k) = xref4dofs4cell[k,:]

get_maxndofs4cell(FE::H1P2FiniteElement{T,1,1} where T <: Real) = 3
get_maxndofs4cell(FE::H1P2FiniteElement{T,2,1} where T <: Real) = 6
get_maxndofs4cell(FE::H1P2FiniteElement{T,2,2} where T <: Real) = 12

get_maxndofs4face(FE::H1P2FiniteElement{T,1,1} where T <: Real) = 1
get_maxndofs4face(FE::H1P2FiniteElement{T,2,1} where T <: Real) = 3
get_maxndofs4face(FE::H1P2FiniteElement{T,2,2} where T <: Real) = 6

# function to get number of components
get_ncomponents(FE::H1P2FiniteElement{T,1,1} where T <: Real) = 1
get_ncomponents(FE::H1P2FiniteElement{T,2,1} where T <: Real) = 1
get_ncomponents(FE::H1P2FiniteElement{T,2,2} where T <: Real) = 2

# function to get global dof numbers of local dofs on cells
get_globaldof4cell(FE::H1P2FiniteElement{T,1,1} where T <: Real, cell, k) = (k <= 2) ? FE.grid.nodes4cells[cell,k] : size(FE.grid.coords4nodes,1) + cell
get_globaldof4cell(FE::H1P2FiniteElement{T,2,1} where T <: Real, cell, k) = (k <= 3) ? FE.grid.nodes4cells[cell,k] : size(FE.grid.coords4nodes,1) + FE.grid.faces4cells[cell,k-3]
function get_globaldof4cell(FE::H1P2FiniteElement{T,2,2} where T <: Real, cell, k)
    if (k <= 3) 
        return FE.grid.nodes4cells[cell,k]
    elseif (k <= 6)
        return size(FE.grid.coords4nodes,1) + FE.grid.faces4cells[cell,k-3]
    elseif (k <= 9) 
        return size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1) + FE.grid.nodes4cells[cell,k-6]
    elseif (k <= 12)
        return 2*size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1) + FE.grid.faces4cells[cell,k-9]
    end    
end

# function to get global dof numbers of local dofs on local face of cell
get_globaldof4cellface(FE::H1P2FiniteElement{T,1,1} where T <: Real, cell, face, k) = FE.grid.nodes4cells[cell,face]
get_globaldof4cellface(FE::H1P2FiniteElement{T,2,1} where T <: Real, cell, face, k) = (k <= 2) ? FE.grid.nodes4faces[cell,k] : size(FE.grid.coords4nodes,1) + FE.grid.faces4cells[cell,face]

# function to get global dof numbers of local dofs on global face
# (only unique due to H1 conformity!!! do not implement for discontinuous elements)
get_globaldof4face(FE::H1P2FiniteElement{T,1,1} where T <: Real, face, k) = face
get_globaldof4face(FE::H1P2FiniteElement{T,2,1} where T <: Real, face, k) = (k <= 2) ? FE.grid.nodes4faces[face,k] : size(FE.grid.coords4nodes,1) + face
function get_globaldof4face(FE::H1P2FiniteElement{T,2,2} where T <: Real, face, k)
    if (k <= 2) 
        return FE.grid.nodes4faces[face,k]
    elseif (k <= 3)
        return size(FE.grid.coords4nodes,1) + face
    elseif (k <= 5) 
        return size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1) + FE.grid.nodes4faces[face,k-3]
    elseif (k <= 6)
        return 2*size(FE.grid.coords4nodes,1) + size(FE.grid.nodes4faces,1) + face
    end    
end


function get_all_basis_functions_on_cell(FE::H1P2FiniteElement{T,1,1} where T <: Real, cell)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1]
        return [2*temp*(temp - 1//2),
                2*xref[1]*(xref[1] - 1//2),
                4*temp*xref[1]]
    end
end

function get_all_basis_functions_on_cell(FE::H1P2FiniteElement{T,2,1} where T <: Real, cell)
    temp = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2]
        return [2*temp*(temp - 1//2),
                2*xref[1]*(xref[1] - 1//2),
                2*xref[2]*(xref[2] - 1//2),
                4*temp*xref[1],
                4*xref[1]*xref[2],
                4*temp*xref[2]]
    end
end


function get_all_basis_functions_on_cell(FE::H1P2FiniteElement{T,2,2} where T <: Real, cell)
    temp = 0.0;
    a = 0.0;
    b = 0.0;
    c = 0.0;
    d = 0.0;
    e = 0.0;
    f = 0.0;
    function closure(xref)
        temp = 1 - xref[1] - xref[2];
        a = 2*temp*(temp - 1//2);
        b = 2*xref[1]*(xref[1] - 1//2);
        c = 2*xref[2]*(xref[2] - 1//2);
        d = 4*temp*xref[1];
        e = 4*xref[1]*xref[2];
        f = 4*temp*xref[2];
        return [a 0.0;    
                b 0.0;
                c 0.0;
                d 0.0;
                e 0.0;
                f 0.0;
                0.0 a;
                0.0 b;
                0.0 c;
                0.0 d;
                0.0 e;
                0.0 f]
                
    end
end
