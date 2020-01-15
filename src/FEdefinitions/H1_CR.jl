struct H1CRFiniteElement{T, dim, ncomponents} <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh{T};           # link to grid
    xref4dofs4cell::Array{T,2};   # coordinates for degrees of freedom in reference domain
end

function getCRFiniteElement(grid,ncomponents)
    ensure_faces4cells!(grid);
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    T = eltype(grid.coords4nodes);
    dim = size(grid.nodes4cells,2) - 1;
    @assert dim == 2
    xref4dofs4cell = repeat([0.5 0; 0.5 0.5; 0 0.5],ncomponents);
    return H1CRFiniteElement{T,dim,ncomponents}("CR (H1FiniteElement, dim=$dim, ncomponents=$ncomponents)",grid, Array{T,2}(xref4dofs4cell))
end   


get_polynomial_order(FE::H1CRFiniteElement) = 1;

get_ndofs(FE::H1CRFiniteElement{T,2,1} where T <: Real) = size(FE.grid.nodes4faces,1);
get_ndofs(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 2*size(FE.grid.nodes4faces,1);

get_xref4dof(FE::H1CRFiniteElement, k) = xref4dofs4cell[k,:]

get_maxndofs4cell(FE::H1CRFiniteElement{T,2,1} where T <: Real) = 3
get_maxndofs4cell(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 6

get_maxndofs4face(FE::H1CRFiniteElement{T,2,1} where T <: Real) = 1
get_maxndofs4face(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 2

# function to get number of components
get_ncomponents(FE::H1CRFiniteElement{T,2,1} where T <: Real) = 1
get_ncomponents(FE::H1CRFiniteElement{T,2,2} where T <: Real) = 2

# function to get global dof numbers of local dofs on cells
get_globaldof4cell(FE::H1CRFiniteElement{T,2,1} where T <: Real, cell, k) = FE.grid.faces4cells[cell,k]

# function to get global dof numbers of local dofs on local face of cell
get_globaldof4cellface(FE::H1CRFiniteElement{T,2,1} where T <: Real, cell, face, k) = FE.grid.faces4cells[cell,face]

# function to get global dof numbers of local dofs on face
# (only unique due to H1 conformity!!! do not implement for discontinuous elements)
get_globaldof4face(FE::H1CRFiniteElement{T,2,1} where T <: Real, face, k) = face

function get_all_basis_functions_on_cell(FE::H1CRFiniteElement{T,2,1} where T <: Real, cell)
    function closure(xref)
        return [1 - 2*xref[2],
                2*(xref[1]+xref[2]) - 1,
                1 - 2*xref[1]]
    end
end
