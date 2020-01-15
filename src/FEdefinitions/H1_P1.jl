struct H1P1FiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getP1FiniteElement(grid,ncomponents)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    if dim == 1
        xref4dofs4cell = repeat(Array{Float64,2}([0,1]')',ncomponents);
    elseif dim == 2    
        xref4dofs4cell = repeat([0 0; 1 0; 0 1],ncomponents);
    end    
    return H1P1FiniteElement{T,dim,ncomponents}("P1 (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end 

get_polynomial_order(FE::H1P1FiniteElement) = 1;

get_ndofs(FE::H1P1FiniteElement{T,1,1} where T <: Real) = size(FE.grid.coords4nodes,1);
get_ndofs(FE::H1P1FiniteElement{T,2,1} where T <: Real) = size(FE.grid.coords4nodes,1);
get_ndofs(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 2*size(FE.grid.coords4nodes,1);

get_xref4dof(FE::H1P1FiniteElement, k) = xref4dofs4cell[k,:]

get_maxndofs4cell(FE::H1P1FiniteElement{T,1,1} where T <: Real) = 2
get_maxndofs4cell(FE::H1P1FiniteElement{T,2,1} where T <: Real) = 3
get_maxndofs4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 6

get_maxndofs4face(FE::H1P1FiniteElement{T,1,1} where T <: Real) = 1
get_maxndofs4face(FE::H1P1FiniteElement{T,2,1} where T <: Real) = 2
get_maxndofs4face(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 4

get_ncomponents(FE::H1P1FiniteElement{T,1,1} where T <: Real) = 1
get_ncomponents(FE::H1P1FiniteElement{T,2,1} where T <: Real) = 1
get_ncomponents(FE::H1P1FiniteElement{T,2,2} where T <: Real) = 2

# function to get global dof numbers of local dofs on cells
get_globaldof4cell(FE::H1P1FiniteElement{T,1,1} where T <: Real, cell, k) = FE.grid.nodes4cells[cell,k]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,1} where T <: Real, cell, k) = FE.grid.nodes4cells[cell,k]
get_globaldof4cell(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, k) = floor(k-1,2)*size(FE.grid.coords4nodes,1) + FE.grid.nodes4cells[cell,(k-1) % 3 + 1];

# function to get global dof numbers of local dofs on local face of cell
get_globaldof4cellface(FE::H1P1FiniteElement{T,1,1} where T <: Real, cell, face, k) = FE.grid.nodes4cells[cell,k]
get_globaldof4cellface(FE::H1P1FiniteElement{T,2,1} where T <: Real, cell, face, k) = FE.grid.nodes4faces[FE.grid.faces4cells[cell,face],k]
get_globaldof4cellface(FE::H1P1FiniteElement{T,2,2} where T <: Real, cell, face, k) = floor(k-1,2)*size(FE.grid.coords4nodes,1) + FE.grid.nodes4faces[FE.grid.faces4cells[cell,face],(k - 1) % 2 + 1]

# function to get global dof numbers of local dofs on face
# (only unique due to H1 conformity!!! do not implement for discontinuous elements)
get_globaldof4face(FE::H1P1FiniteElement{T,1,1} where T <: Real, face, k) = face
get_globaldof4face(FE::H1P1FiniteElement{T,2,1} where T <: Real, face, k) = FE.grid.nodes4faces[face,k]
get_globaldof4face(FE::H1P1FiniteElement{T,2,2} where T <: Real, face, k) = floor(k-1,2)*size(FE.grid.coords4nodes,1) + FE.grid.nodes4faces[face,(k - 1) % 2]


function get_all_basis_functions_on_cell(FE::H1P1FiniteElement{T,1,1} where T <: Real, cell)
    function closure(xref)
        return [1 - xref[1],
                xref[1]]
    end
end

function get_all_basis_functions_on_cell(FE::H1P1FiniteElement{T,2,1} where T <: Real, cell)
    function closure(xref)
        return [1 - xref[1] - xref[2],
                xref[1],
                xref[2]]
    end
end
