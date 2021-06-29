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
    offsetX::Int64
    offsetY::Int64
    last_indexX::Int64
    last_indexY::Int64
    entries::AbstractMatrix # shares with parent object
end

"""
$(TYPEDEF)

an AbstractMatrix (e.g. an ExtendableSparseMatrix) with an additional layer of several FEMatrixBlock subdivisions each carrying coefficients for their associated pair of FESpaces
"""
struct FEMatrix{T,nbrow,nbcol,nbtotal}
    FEMatrixBlocks::Array{FEMatrixBlock{T},1}
    entries::AbstractMatrix
end

#Add value to matrix if it is nonzero
@inline function _addnz(ESM::ExtendableSparseMatrix,i,j,v::Tv,fac) where Tv
    if v!=zero(Tv)
        rawupdateindex!(ESM,+,v*fac,i,j)
    end
end

#Add value to matrix if it is nonzero
@inline function _addnz(FEB::FEMatrixBlock,i,j,v::Tv,fac) where Tv
    if v!=zero(Tv)
       rawupdateindex!(FEB.entries,+,v*fac,FEB.offsetX+i,FEB.offsetY+j)
    end
end

function apply_nonzero_pattern!(B::FEMatrixBlock,AT::Type{<:AbstractAssemblyType})
    dofmapX = Dofmap4AssemblyType(B.FESX,AT)
    dofmapY = Dofmap4AssemblyType(B.FESY,AT)
    @assert num_sources(dofmapX) == num_sources(dofmapY)
    for item = 1 : num_sources(dofmapX)
        for j = 1 : num_targets(dofmapX,item), k = 1 : num_targets(dofmapY,item)
            rawupdateindex!(B.entries,+,0,B.offsetX+dofmapX[j,item],B.offsetY+dofmapY[k,item])
        end
    end
end

Base.getindex(FEF::FEMatrix,i) = FEF.FEMatrixBlocks[i]
Base.getindex(FEF::FEMatrix{T,nbrow,nbcol},i,j) where {T,nbrow,nbcol} = FEF.FEMatrixBlocks[(i-1)*nbcol+j]
Base.getindex(FEB::FEMatrixBlock,i::Int,j::Int)=FEB.entries[FEB.offsetX+i,FEB.offsetY+j]
Base.getindex(FEB::FEMatrixBlock,i::Any,j::Any)=FEB.entries[FEB.offsetX.+i,FEB.offsetY.+j]
Base.setindex!(FEB::FEMatrixBlock, v, i::Int, j::Int) = (FEB.entries[FEB.offsetX+i,FEB.offsetY+j] = v)

"""
$(TYPEDSIGNATURES)

Custom `size` function for `FEMatrix` that gives the number of rows and columns of the FEBlock overlay
"""
Base.size(FEF::FEMatrix) = [typeof(FEF).parameters[2],typeof(FEF).parameters[3]]


"""
$(TYPEDSIGNATURES)

Custom `length` function for `FEMatrix` that gives the total number of defined FEMatrixBlocks in it
"""
Base.length(FEF::FEMatrix) = typeof(FEF).parameters[4]

"""
$(TYPEDSIGNATURES)

Custom `size` function for `FEMatrixBlock` that gives the size of the block (that coressponds to the number of degrees of freedoms in X and Y)
"""
Base.size(FEB::FEMatrixBlock)=[FEB.last_indexX-FEB.offsetX,FEB.last_indexY-FEB.offsetY]

"""
$(TYPEDSIGNATURES)

Custom `show` function for `FEMatrix` that prints some information on its blocks.
"""
function Base.show(io::IO, FEM::FEMatrix{T,nbrow}) where {T,nbrow}
	println("\nFEMatrix information")
    println("====================")
    println("  block |  ndofsX |  ndofsY | name (FETypeX, FETypeY) ")
    for j=1:length(FEM)
        n = mod(j-1,nbrow)+1
        m = Int(ceil(j/nbrow))
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
    entries = ExtendableSparseMatrix{T,Int64}(FES.ndofs,FES.ndofs)
    Block = FEMatrixBlock{T}(name, FES, FES, 0 , 0, FES.ndofs, FES.ndofs, entries)
    return FEMatrix{T,1,1,1}([Block], entries)
end

"""
````
FEMatrix{T}(name::String, FESX::FESpace, FESY::FESpace) where T <: Real
````

Creates FEMatrix with one rectangular block (FESX,FESY).
"""
function FEMatrix{T}(name::String, FESX::FESpace, FESY::FESpace) where T <: Real
    entries = ExtendableSparseMatrix{T,Int64}(FESX.ndofs,FESY.ndofs)
    Block = FEMatrixBlock{T}(name, FESX, FESY, 0 , 0, FESX.ndofs, FESY.ndofs, entries)
    return FEMatrix{T,1,1,1}([Block], entries)
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
    entries = ExtendableSparseMatrix{T,Int64}(ndofs,ndofs)

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
    
    return FEMatrix{T,length(FES),length(FES),length(FES)^2}(Blocks, entries)
end

"""
$(TYPEDSIGNATURES)

Custom `fill` function for `FEMatrixBlock` (only fills the block, not the complete FEMatrix).
"""
function Base.fill!(B::FEMatrixBlock, value)
    cscmat::SparseMatrixCSC{Float64,Int64} = B.entries.cscmatrix
    rows::Array{Int,1} = rowvals(cscmat)
    valsB::Array{Float64,1} = cscmat.nzval
    for col = B.offsetY+1:B.last_indexY
        for r in nzrange(cscmat, col)
            if rows[r] > B.offsetX && rows[r] <= B.last_indexX
                valsB[r] = value
            end
        end
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Adds FEMatrixBlock B to FEMatrixBlock A.
"""
function addblock!(A::FEMatrixBlock, B::FEMatrixBlock; factor = 1, transpose::Bool = false)
    cscmat::SparseMatrixCSC{Float64,Int64} = B.entries.cscmatrix
    rows::Array{Int,1} = rowvals(cscmat)
    valsB::Array{Float64,1} = cscmat.nzval
    arow::Int = 0
    acol::Int = 0
    if transpose
        for col = B.offsetY+1:B.last_indexY
            arow = col - B.offsetY + A.offsetX
            for r in nzrange(cscmat, col)
                if rows[r] >= B.offsetX && rows[r] <= B.last_indexX
                    acol = rows[r] - B.offsetX + A.offsetY
                    _addnz(A.entries,arow,acol,valsB[r],factor)
                end
            end
        end
    else
        for col = B.offsetY+1:B.last_indexY
            acol = col - B.offsetY + A.offsetY
            for r in nzrange(cscmat, col)
                if rows[r] >= B.offsetX && rows[r] <= B.last_indexX
                    arow = rows[r] - B.offsetX + A.offsetX
                    _addnz(A.entries,arow,acol,valsB[r],factor)
                end
            end
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Adds ExtendableSparseMatrix B to FEMatrixBlock A.
"""
function addblock!(A::FEMatrixBlock, B::ExtendableSparseMatrix{Tv,Ti}; factor = 1, transpose::Bool = false) where {Tv, Ti <: Integer}
    cscmat::SparseMatrixCSC{Tv, Ti} = B.cscmatrix
    rows::Array{Int,1} = rowvals(cscmat)
    valsB::Array{Tv,1} = cscmat.nzval
    arow::Int = 0
    acol::Int = 0
    if transpose
        for col = 1:size(B,2)
            arow = col + A.offsetX
            for r in nzrange(cscmat, col)
                acol = rows[r] + A.offsetY
                _addnz(A.entries,arow,acol,valsB[r],factor)
                #A[rows[r],col] += B.cscmatrix.nzval[r] * factor
            end
        end
    else
        for col = 1:size(B,2)
            acol = col + A.offsetY
            for r in nzrange(cscmat, col)
                arow = rows[r] + A.offsetX
                _addnz(A.entries,arow,acol,valsB[r],factor)
                #A[rows[r],col] += B.cscmatrix.nzval[r] * factor
            end
        end
    end
    return nothing
end


function addblock!(A::ExtendableSparseMatrix{Tv,Ti}, B::ExtendableSparseMatrix{Tv,Ti}; factor = 1, transpose::Bool = false) where {Tv, Ti <: Integer}
    cscmat::SparseMatrixCSC{Tv, Ti} = B.cscmatrix
    rows::Array{Int,1} = rowvals(cscmat)
    valsB::Array{Tv,1} = cscmat.nzval
    arow::Int = 0
    acol::Int = 0
    if transpose
        for col = 1:size(B,2)
            arow = col
            for r in nzrange(cscmat, col)
                acol = rows[r]
                _addnz(A,arow,acol,valsB[r],factor)
                #A[rows[r],col] += B.cscmatrix.nzval[r] * factor
            end
        end
    else
        for col = 1:size(B,2)
            acol = col
            for r in nzrange(cscmat, col)
                arow = rows[r]
                _addnz(A,arow,acol,valsB[r],factor)
                #A[rows[r],col] += B.cscmatrix.nzval[r] * factor
            end
        end
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Adds matrix-vector product B times b to FEVectorBlock a.
"""
function addblock_matmul!(a::FEVectorBlock, B::FEMatrixBlock, b::FEVectorBlock; factor = 1, transposed::Bool = false)
    cscmat::SparseMatrixCSC{Float64,Int64} = B.entries.cscmatrix
    rows::Array{Int64,1} = rowvals(cscmat)
    valsB::Array{Float64,1} = cscmat.nzval
    bcol::Int = 0
    row::Int = 0
    arow::Int = 0
    if transposed
        for col = B.offsetY+1:B.last_indexY
            bcol = col-B.offsetY+a.offset
            for r in nzrange(cscmat, col)
                row = rows[r]
                if row >= B.offsetX && row <= B.last_indexX
                    arow = row - B.offsetX + b.offset
                    a.entries[bcol] += valsB[r] * b.entries[arow] * factor 
                end
            end
        end
    else
        for col = B.offsetY+1:B.last_indexY
            bcol = col-B.offsetY+b.offset
            for r in nzrange(cscmat, col)
                row = rows[r]
                if row >= B.offsetX && row <= B.last_indexX
                    arow = row - B.offsetX + a.offset
                    a.entries[arow] += valsB[r] * b.entries[bcol] * factor 
                end
            end
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Adds matrix-vector product B times b to FEVectorBlock a.
"""
function addblock_matmul!(a::AbstractVector, B::FEMatrixBlock, b::AbstractVector; factor = 1, transposed::Bool = false)
    cscmat::SparseMatrixCSC{Float64,Int64} = B.entries.cscmatrix
    rows::Array{Int64,1} = rowvals(cscmat)
    valsB::Array{Float64,1} = cscmat.nzval
    bcol::Int = 0
    row::Int = 0
    arow::Int = 0
    if transposed
        for col = B.offsetY+1:B.last_indexY
            bcol = col-B.offsetY
            for r in nzrange(cscmat, col)
                row = rows[r]
                if row >= B.offsetX && row <= B.last_indexX
                    arow = row - B.offsetX
                    a[bcol] += valsB[r] * b[arow] * factor 
                end
            end
        end
    else
        for col = B.offsetY+1:B.last_indexY
            bcol = col-B.offsetY
            for r in nzrange(cscmat, col)
                row = rows[r]
                if row >= B.offsetX && row <= B.last_indexX
                    arow = row - B.offsetX
                    a[arow] += valsB[r] * b[bcol] * factor 
                end
            end
        end
    end
    return nothing
end



"""
$(TYPEDSIGNATURES)

Adds matrix-vector product B times b to FEVectorBlock a.
"""
function addblock_matmul!(a::FEVectorBlock, B::ExtendableSparseMatrix{Tv,Ti}, b::FEVectorBlock; factor = 1) where {Tv, Ti <: Integer}
    cscmat::SparseMatrixCSC{Tv,Ti} = B.cscmatrix
    rows::Array{Ti,1} = rowvals(cscmat)
    valsB::Array{Tv,1} = cscmat.nzval
    bcol::Int = 0
    arow::Int = 0
    for col = 1:size(B,2)
        bcol = col+b.offset
        for r in nzrange(cscmat, col)
            arow = rows[r] + a.offset
            a.entries[arow] += valsB[r] * b.entries[bcol] * factor
        end
    end
    return nothing
end



"""
$(TYPEDSIGNATURES)

Computes vector'-matrix-vector product a'*B*b.
"""
function lrmatmul(a::AbstractVector{Tv}, B::ExtendableSparseMatrix{Tv,Ti}, b::AbstractVector{Tv}; factor = 1) where {Tv, Ti <: Integer}
    cscmat::SparseMatrixCSC{Tv,Ti} = B.cscmatrix
    valsB::Array{Tv,1} = cscmat.nzval
    rows::Array{Ti,1} = rowvals(cscmat)
    result = 0.0
    for col = 1:size(B,2)
        for r in nzrange(B.cscmatrix, col)
            result += valsB[r] * b[col] * factor * a[rows[r]]
        end
    end
    return result
end


"""
$(TYPEDSIGNATURES)

Computes vector'-matrix-vector product (a1-a2)'*B*(b1-b2).
"""
function ldrdmatmul(a1::AbstractVector{Tv}, a2::AbstractVector{Tv}, B::ExtendableSparseMatrix{Tv,Ti}, b1::AbstractVector{Tv}, b2::AbstractVector{Tv}; factor = 1) where {Tv, Ti <: Integer}
    cscmat::SparseMatrixCSC{Tv,Ti} = B.cscmatrix
    valsB::Array{Tv,1} = cscmat.nzval
    rows::Array{Ti,1} = rowvals(cscmat)
    result = 0.0
    for col = 1:size(B,2)
        for r in nzrange(B.cscmatrix, col)
            result += valsB[r] * (b1[col] - b2[col]) * factor * (a1[rows[r]] - a2[rows[r]])
        end
    end
    return result
end