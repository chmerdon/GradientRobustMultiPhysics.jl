
mutable struct L2GTransformer{T <: Real, EG <: AbstractElementGeometry, CS <: AbstractCoordinateSystem}
    citem::Int
    nonlinear::Bool
    Coords::Array{T,2}
    Nodes::Union{VariableTargetAdjacency{Int32},Array{Int32,2}}
    ItemVolumes::Array{T,1}
    A::Matrix{T}
    b::Vector{T}
end    

function L2GTransformer{T,EG,CS}(grid::ExtendableGrid, AT::Type{<:AbstractAssemblyType})  where {T <: Real, EG <: AbstractElementGeometry, CS <: AbstractCoordinateSystem}
    A = zeros(T,size(grid[Coordinates],1),dim_element(EG))
    b = zeros(T,size(grid[Coordinates],1))
    return L2GTransformer{T,EG,CS}(0,false,grid[Coordinates],grid[GridComponentNodes4AssemblyType(AT)],grid[GridComponentVolumes4AssemblyType(AT)],A,b)
end



function update!(T::L2GTransformer{<:Real,<:Edge1D,Cartesian2D}, item::Int)
    if T.citem != item
        T.citem = item
        T.b[1] = T.Coords[1,T.Nodes[1,item]]
        T.b[2] = T.Coords[2,T.Nodes[1,item]]
        T.A[1,1] = T.Coords[1,T.Nodes[2,item]] - T.b[1]
        T.A[2,1] = T.Coords[2,T.Nodes[2,item]] - T.b[2]
    end    
end

function update!(T::L2GTransformer{<:Real,<:Triangle2D,Cartesian2D}, item::Int)
    if T.citem != item
        T.citem = item
        T.b[1] = T.Coords[1,T.Nodes[1,item]]
        T.b[2] = T.Coords[2,T.Nodes[1,item]]
        T.A[1,1] = T.Coords[1,T.Nodes[2,item]] - T.b[1]
        T.A[1,2] = T.Coords[1,T.Nodes[3,item]] - T.b[1]
        T.A[2,1] = T.Coords[2,T.Nodes[2,item]] - T.b[2]
        T.A[2,2] = T.Coords[2,T.Nodes[3,item]] - T.b[2]
    end    
end

function update!(T::L2GTransformer{<:Real,<:Parallelogram2D,Cartesian2D}, item::Int)
    if T.citem != item
        T.citem = item
        T.b[1] = T.Coords[1,T.Nodes[1,item]]
        T.b[2] = T.Coords[2,T.Nodes[1,item]]
        T.A[1,1] = T.Coords[1,T.Nodes[2,item]] - T.b[1]
        T.A[1,2] = T.Coords[1,T.Nodes[4,item]] - T.b[1]
        T.A[2,1] = T.Coords[2,T.Nodes[2,item]] - T.b[2]
        T.A[2,2] = T.Coords[2,T.Nodes[4,item]] - T.b[2]
    end    
end

function eval!(x::Vector, T::L2GTransformer{<:Real,<:Union{Triangle2D, Parallelogram2D},Cartesian2D}, xref)
    x[1] = T.A[1,1]*xref[1] + T.A[1,2]*xref[2] + T.b[1]
    x[2] = T.A[2,1]*xref[1] + T.A[2,2]*xref[2] + T.b[2]
end


function eval!(x::Vector, T::L2GTransformer{<:Real,<:Edge1D,Cartesian2D}, xref)
    x[1] = T.A[1,1]*xref[1] + T.b[1]
    x[2] = T.A[2,1]*xref[1] + T.b[2]
end

# TRIANGLE2D/CARTESIAN2D map derivative
# x = A*xref + b
# Dxref/dx = A^{-T}
function mapderiv!(M::Matrix, T::L2GTransformer{<:Real,<:Triangle2D,Cartesian2D}, xref)
    # transposed inverse of A
    det = 2*T.ItemVolumes[T.citem]
    M[2,2] = T.A[1,1]/det
    M[2,1] = -T.A[1,2]/det
    M[1,2] = -T.A[2,1]/det
    M[1,1] = T.A[2,2]/det
    return det
end
# similar for parallelogram
function mapderiv!(M::Matrix, T::L2GTransformer{<:Real,<:Parallelogram2D,Cartesian2D}, xref)
    # transposed inverse of A
    det = T.ItemVolumes[T.citem]
    M[2,2] = T.A[1,1]/det
    M[2,1] = -T.A[1,2]/det
    M[1,2] = -T.A[2,1]/det
    M[1,1] = T.A[2,2]/det
    return det
end

# TRIANGLE2D/CARTESIAN2D Piola map
# x = A*xref + b
# returns A
function piola!(M::Matrix, T::L2GTransformer{<:Real,<:Triangle2D,Cartesian2D}, xref)
    M[1,1] = T.A[1,1]
    M[1,2] = T.A[1,2]
    M[2,1] = T.A[2,1]
    M[2,2] = T.A[2,2]
    return 2*T.ItemVolumes[T.citem]
end
# similar for parallelogram
function piola!(M::Matrix, T::L2GTransformer{<:Real,<:Parallelogram2D,Cartesian2D}, xref)
    M[1,1] = T.A[1,1]
    M[1,2] = T.A[1,2]
    M[2,1] = T.A[2,1]
    M[2,2] = T.A[2,2]
    return T.ItemVolumes[T.citem]
end