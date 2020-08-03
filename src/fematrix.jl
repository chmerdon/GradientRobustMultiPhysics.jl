################
### FEMatrix ###
################
#
# used to store (sparse) matrix representations of PDEOperators for FESpaces
# and can have several blocks of different FESpaces
# acts like an AbstractArray{T,2}

"""
$(TYPEDEF)

block of an FEMatrix that carries coefficients for an associated pair of FESpaces and can be assigned as an two-dimensional AbstractArray (getindex, setindex, size)
"""
struct FEMatrixBlock{T} <: AbstractArray{T,2}
    name::String
    FESX::FESpace
    FESY::FESpace
    offsetX::Int32
    offsetY::Int32
    last_indexX::Int32
    last_indexY::Int32
    entries::AbstractMatrix # shares with parent object
end

"""
$(TYPEDEF)

an AbstractMatrix (e.g. an ExtendableSparseMatrix) with an additional layer of several FEMatrixBlock subdivisions each carrying coefficients for their associated pair of FESpaces
"""
struct FEMatrix{T} <: AbstractArray{T,1}
    FEMatrixBlocks::Array{FEMatrixBlock{T},1}
    nFE::Int
    entries::AbstractMatrix
end

Base.getindex(FEF::FEMatrix,i) = FEF.FEMatrixBlocks[i]
Base.getindex(FEF::FEMatrix,i,j) = FEF.FEMatrixBlocks[(i-1)*FEF.nFE+j]
Base.getindex(FEB::FEMatrixBlock,i::Int,j::Int)=FEB.entries[FEB.offsetX+i,FEB.offsetY+j]
Base.getindex(FEB::FEMatrixBlock,i::Any,j::Any)=FEB.entries[FEB.offsetX.+i,FEB.offsetY.+j]
Base.setindex!(FEB::FEMatrixBlock, v, i::Int, j::Int) = (FEB.entries[FEB.offsetX+i,FEB.offsetY+j] = v)
Base.size(FEF::FEMatrix)=length(FEF.FEMatrixBlocks)
Base.size(FEB::FEMatrixBlock)=[FEB.last_indexX-FEB.offsetX,FEB.last_indexY-FEB.offsetY]

"""
$(TYPEDSIGNATURES)

Custom `show` function for `FEMatrix` that prints some information on its blocks.
"""
function Base.show(io::IO, FEM::FEMatrix)
	println("\nFEMatrix information")
    println("====================")
    println("  block |  ndofsX |  ndofsY | name (FETypeX, FETypeY) ")
    for j=1:length(FEM)
        n = mod(j-1,FEM.nFE)+1
        m = Int(ceil(j/FEM.nFE))
        @printf("  [%d,%d] |",m,n);
        @printf("  %6d |",FEM[j].FESX.ndofs);
        @printf("  %6d |",FEM[j].FESY.ndofs);
        @printf(" %s (%s,%s)\n",FEM[j].name,FEM[j].FESX.name,FEM[j].FESY.name);
    end    
end

"""
````
FEMatrix{T}(name::String, FES::FESpace) where T <: Real
````

Creates FEMatrix with one square block (FES,FES).
"""
function FEMatrix{T}(name::String, FES::FESpace) where T <: Real
    entries = ExtendableSparseMatrix{T,Int32}(FES.ndofs,FES.ndofs)
    Block = FEMatrixBlock{T}(name, FES, FES, 0 , 0, FES.ndofs, FES.ndofs, entries)
    return FEMatrix{T}([Block], 1, entries)
end

"""
````
FEMatrix{T}(name::String, FESX::FESpace, FESY::FESpace) where T <: Real
````

Creates FEMatrix with one rectangular block (FESX,FESY).
"""
function FEMatrix{T}(name::String, FESX::FESpace, FESY::FESpace) where T <: Real
    entries = ExtendableSparseMatrix{T,Int32}(FESX.ndofs,FESY.ndofs)
    Block = FEMatrixBlock{T}(name, FESX, FESY, 0 , 0, FESX.ndofs, FESY.ndofs, entries)
    return FEMatrix{T}([Block], 2, entries)
end

"""
````
FEMatrix{T}(name::String, FES::Array{FESpace,1}) where T <: Real
````

Creates FEMatrix with blocks (FESX[i],FESY[j]) (enumerated row-wise).
"""
function FEMatrix{T}(name::String, FES::Array{FESpace,1}) where T <: Real
    ndofs = 0
    for j=1:length(FES)
        ndofs += FES[j].ndofs
    end
    entries = ExtendableSparseMatrix{T,Int32}(ndofs,ndofs)

    Blocks = Array{FEMatrixBlock{T},1}(undef,length(FES)^2)
    offsetX = 0
    offsetY = 0
    for j=1:length(FES)
        offsetY = 0
        for k=1:length(FES)
            Blocks[(j-1)*length(FES)+k] = FEMatrixBlock{T}(name * " [$j,$k]", FES[j], FES[k], offsetX , offsetY, offsetX+FES[j].ndofs, offsetY+FES[k].ndofs, entries)
            offsetY += FES[k].ndofs
        end    
        offsetX += FES[j].ndofs
    end    
    
    return FEMatrix{T}(Blocks, length(FES), entries)
end

"""
$(TYPEDSIGNATURES)

Custom `fill` function for `FEMatrixBlock` (only fills the block, not the complete FEMatrix).
"""
function Base.fill!(B::FEMatrixBlock, value::Real)
    rows = rowvals(B.entries.cscmatrix)
    for col = B.offsetY+1:B.last_indexY
        for r in nzrange(B.entries.cscmatrix, col)
            if rows[r] >= B.offsetX && rows[r] <= B.last_indexX
                B.entries.cscmatrix.nzval[r] = value
            end
        end
    end
end

"""
$(TYPEDSIGNATURES)

Adds FEMatrixBlock B to FEMatrixBlock A.
"""
function addblock!(A::FEMatrixBlock, B::FEMatrixBlock; factor::Real = 1)
    rows = rowvals(B.entries.cscmatrix)
    for col = B.offsetY+1:B.last_indexY
        for r in nzrange(B.entries.cscmatrix, col)
            if rows[r] >= B.offsetX && rows[r] <= B.last_indexX
                A[rows[r]-B.offsetX,col-B.offsetY] += B.entries.cscmatrix.nzval[r] * factor
            end
        end
    end
end

"""
$(TYPEDSIGNATURES)

Adds ExtendableSparseMatrix B to FEMatrixBlock A.
"""
function addblock!(A::FEMatrixBlock, B::ExtendableSparseMatrix; factor::Real = 1)
    rows = rowvals(B.cscmatrix)
    for col = 1:size(B,2)
        for r in nzrange(B.cscmatrix, col)
            A[rows[r],col] += B.cscmatrix.nzval[r] * factor
        end
    end
end

"""
$(TYPEDSIGNATURES)

Adds matrix-vector product B times b to FEVectorBlock a.
"""
function addblock_matmul!(a::FEVectorBlock, B::FEMatrixBlock, b::FEVectorBlock; factor::Real = 1)
    rows = rowvals(B.entries.cscmatrix)
    for col = B.offsetY+1:B.last_indexY
        for r in nzrange(B.entries.cscmatrix, col)
            if rows[r] >= B.offsetX && rows[r] <= B.last_indexX
                a[rows[r]-B.offsetX] += B.entries.cscmatrix.nzval[r] * b[col-B.offsetY] * factor 
            end
        end
    end
end



"""
$(TYPEDSIGNATURES)

Adds matrix-vector product B times b to FEVectorBlock a.
"""
function addblock_matmul!(a::FEVectorBlock, B::ExtendableSparseMatrix, b::FEVectorBlock; factor::Real = 1)
    rows = rowvals(B.cscmatrix)
    for col = 1:size(B,2)
        for r in nzrange(B.cscmatrix, col)
            a[rows[r]] += B.cscmatrix.nzval[r] * b[col] * factor
        end
    end
end