struct H1CompositeFiniteElement <: AbstractH1FiniteElement
    name::String;                 # full name of finite element (used in messages)
    grid::Grid.Mesh;           # link to grid
    FEs::Array{AbstractH1FiniteElement,1}
    maxpoly_order::Int64
    ndofs::Array{Int64,1}
end

function string2FE(FElabels::Array{String,1}, grid::Grid.Mesh, dim::Int)
    FEs = Array{AbstractH1FiniteElement,1}(undef,length(FElabels))
    label::String = ""
    maxpolyorder::Int64 = 0
    ndofs = zeros(Int64,length(FElabels)+1)
    for j=1:length(FElabels)
        FEs[j] = string2FE(FElabels[j], grid, dim, 1)
        pos = findfirst(isequal(' '), FEs[j].name)
        label *= SubString(FEs[j].name,1,pos-1)
        if j < length(FElabels)
            label *= " x "
        end    
        maxpolyorder = max(maxpolyorder,get_polynomial_order(FEs[j]));
        ndofs[j+1] = ndofs[j] + get_ndofs(FEs[j])
    end
    label *= " (H1CompositeFiniteElement, dim=$dim)"
    return H1CompositeFiniteElement(label,grid,FEs,maxpolyorder,ndofs)
end   

function get_xref4dof(CFE::H1CompositeFiniteElement, ET::Grid.AbstractElemType) 
    xref = []
    InterpolationMatrix = []
    for j=1:length(CFE.FEs)
        xref_add, Interpolationmatrix_add = get_xref4dof(CFE.FEs[j], ET)
        append!(xref,xref_add)
        append!(InterpolationMatrix,Interpolationmatrix_add)
    end    
    return xref, InterpolationMatrix
end    

# POLYNOMIAL ORDER
get_polynomial_order(CFE::H1CompositeFiniteElement) = CFE.maxpoly_order;

# TOTAL NUMBER OF DOFS
get_ndofs(CFE::H1CompositeFiniteElement) = CFE.ndofs[end];

# NUMBER OF DOFS ON ELEMTYPE
function get_ndofs4elemtype(CFE::H1CompositeFiniteElement, ET::Grid.AbstractElemType)
    ndofs::Int64 = 0
    for j=1:length(CFE.FEs)
        ndofs += get_ndofs4elemtype(CFE.FEs[j], ET)
    end
    return ndofs
end

# NUMBER OF COMPONENTS
get_ncomponents(CFE::H1CompositeFiniteElement) = length(CFE.FEs);


# LOCAL DOF TO GLOBAL DOF ON CELL
function get_dofs_on_cell!(dofs, CFE::H1CompositeFiniteElement, cell::Int64, ET::Grid.AbstractElemType)
    ndofs::Int64 = 0
    offset::Int64 = 0
    for j=1:length(CFE.FEs)
        ndofs = get_ndofs4elemtype(CFE.FEs[j], ET)
        get_dofs_on_cell!(view(dofs,offset+1:offset+ndofs),CFE.FEs[j],cell,ET)
        dofs[offset+1:offset+ndofs] .+= CFE.ndofs[j]
        offset += ndofs
    end
end

# LOCAL DOF TO GLOBAL DOF ON FACE
function get_dofs_on_face!(dofs,CFE::H1CompositeFiniteElement, face::Int64, ET::Grid.AbstractElemType)
    ndofs::Int64 = 0
    offset::Int64 = 0
    for j=1:length(CFE.FEs)
        ndofs = get_ndofs4elemtype(CFE.FEs[j], ET)
        get_dofs_on_face!(view(dofs,offset+1:offset+ndofs),CFE.FEs[j],face,ET)
        dofs[offset+1:offset+ndofs] .+= CFE.ndofs[j]
        offset += ndofs
    end
end

# BASIS FUNCTIONS
function get_basis_on_cell(CFE::H1CompositeFiniteElement, ET::Grid.AbstractElemType)
    ndofs::Int64 = get_ndofs4elemtype(CFE, ET)
    basis = zeros(Real,ndofs,length(CFE.FEs))
    offset::Int64 = 0
    function closure(xref)
        ndofs = 0
        offset = 0
        for j=1:length(CFE.FEs)
            ndofs = get_ndofs4elemtype(CFE.FEs[j], ET)
            basis[offset+1:offset+ndofs,j] = get_basis_on_cell(CFE.FEs[j],ET)(xref)
            offset += ndofs
        end
        return basis
    end
end

function get_basis_on_face(CFE::H1CompositeFiniteElement, ET::Grid.AbstractElemType)
    ndofs::Int64 = get_ndofs4elemtype(CFE, ET)
    offset::Int64 = 0
    basis = zeros(Real,ndofs,length(CFE.FEs))
    function closure(xref)
        ndofs = 0
        offset = 0
        for j=1:length(CFE.FEs)
            ndofs = get_ndofs4elemtype(CFE.FEs[j], ET)
            basis[offset+1:offset+ndofs,j] = get_basis_on_face(CFE.FEs[j],ET)(xref)
            offset += ndofs
        end
        return basis
    end
end