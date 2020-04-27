
mutable struct L2GTransformer{T <: Real, EG <: AbstractElementGeometry, CS <: XGrid.AbstractCoordinateSystem}
    current_item::Int
    Coords::Array{T,2}
    Nodes::Union{VariableTargetAdjacency{Int32},Array{Int32,2}}
    A::Matrix{T}
    b::Vector{T}
end    

function L2GTransformer{T,EG,CS}(grid::ExtendableGrid, AT::Type{<:AbstractAssemblyType})  where {T <: Real, EG <: AbstractElementGeometry, CS <: XGrid.AbstractCoordinateSystem}
    A = zeros(T,2,2)
    b = zeros(T,2)
    return L2GTransformer{T,EG,CS}(0,grid[Coordinates],grid[GridComponentNodes4AssemblyType(AT)],A,b)
end


function update!(T::L2GTransformer{<:Real,<:Edge1D,Cartesian2D}, item::Int)
    T.current_item = item
    T.b[1] = T.Coords[1,T.Nodes[1,item]]
    T.b[2] = T.Coords[2,T.Nodes[1,item]]
    T.A[1,1] = T.Coords[1,T.Nodes[2,item]] - T.b[1]
    T.A[2,1] = T.Coords[2,T.Nodes[2,item]] - T.b[2]
end

function update!(T::L2GTransformer{<:Real,<:Triangle2D,Cartesian2D}, item::Int)
    T.current_item = item
    T.b[1] = T.Coords[1,T.Nodes[1,item]]
    T.b[2] = T.Coords[2,T.Nodes[1,item]]
    T.A[1,1] = T.Coords[1,T.Nodes[2,item]] - T.b[1]
    T.A[1,2] = T.Coords[1,T.Nodes[3,item]] - T.b[1]
    T.A[2,1] = T.Coords[2,T.Nodes[2,item]] - T.b[2]
    T.A[2,2] = T.Coords[2,T.Nodes[3,item]] - T.b[2]
end

function update!(T::L2GTransformer{<:Real,<:Parallelogram2D,Cartesian2D}, item::Int)
    T.current_item = item
    T.b[1] = T.Coords[1,T.Nodes[1,item]]
    T.b[2] = T.Coords[2,T.Nodes[1,item]]
    T.A[1,1] = T.Coords[1,T.Nodes[2,item]] - T.b[1]
    T.A[1,2] = T.Coords[1,T.Nodes[4,item]] - T.b[1]
    T.A[2,1] = T.Coords[2,T.Nodes[2,item]] - T.b[2]
    T.A[2,2] = T.Coords[2,T.Nodes[4,item]] - T.b[2]
end

function eval!(x::Vector, T::L2GTransformer{<:Real,<:Union{Triangle2D, Parallelogram2D},Cartesian2D}, xref)
    x[1] = T.A[1,1]*xref[1] + T.A[1,2]*xref[2] + T.b[1]
    x[2] = T.A[2,1]*xref[1] + T.A[2,2]*xref[2] + T.b[2]
end


function eval!(x::Vector, T::L2GTransformer{<:Real,<:Union{Edge1D},Cartesian2D}, xref)
    x[1] = T.A[1,1]*xref[1] + T.b[1]
    x[2] = T.A[2,1]*xref[1] + T.b[2]
end