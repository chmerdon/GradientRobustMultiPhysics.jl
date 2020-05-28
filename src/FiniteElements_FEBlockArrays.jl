################
### FEVector ###
################
#
# used to store coefficients for FESpaces and can have several blocks of different FESpaces
# acts like an AbstractArray{T,1}

struct FEVectorBlock{T} <: AbstractArray{T,1}
    name::String
    FES::FESpace
    offset::Int
    last_index::Int
    entries::Array{T,1} # shares with parent object
end

struct FEVector{T} <: AbstractArray{T,1}
    FEVectorBlocks::Array{FEVectorBlock{T},1}
    entries::Array{T,1}
end

function FEVector{T}(name::String, FES::FESpace) where T <: Real
    entries = zeros(T,FES.ndofs)
    Block = FEVectorBlock{T}(name, FES, 0 , size(entries,1), entries)
    return FEVector{T}([Block], entries)
end

function FEVector{T}(name::String, FES::Array{FESpace,1}) where T <: Real
    ndofs = 0
    for j = 1:length(FES)
        ndofs += FES[j].ndofs
    end    
    entries = zeros(T,ndofs)
    Blocks = Array{FEVectorBlock,1}(undef,length(FES))
    offset = 0
    for j = 1:length(FES)
        Blocks[j] = FEVectorBlock{T}(name, FES[j], offset , offset+FES[j].ndofs, entries)
        offset += FES[j].ndofs
    end    
    return FEVector{T}(Blocks, entries)
end

function Base.show(io::IO, FEF::FEVector) where {T}
	println("\nFEVector information")
    println("====================")
    println("   block  |  ndofs  | name (FEType) ")
    for j=1:length(FEF)
        @printf(" [%5d]  | ",j);
        @printf(" %6d |",FEF[j].FES.ndofs);
        @printf(" %s (%s)\n",FEF[j].name,FEF[j].FES.name);
    end    
end

# overload stuff for AbstractArray{T,1} behaviour
Base.getindex(FEF::FEVector,i) = FEF.FEVectorBlocks[i]
Base.getindex(FEB::FEVectorBlock,i::Int)=FEB.entries[FEB.offset+i]
Base.getindex(FEB::FEVectorBlock,i::AbstractArray)=FEB.entries[FEB.offset.+i]
Base.getindex(FEB::FEVectorBlock,::Colon)=FEB.entries[FEB.offset+1:FEB.last_index]
Base.setindex!(FEB::FEVectorBlock, v, i::Int) = (FEB.entries[FEB.offset+i] = v)
Base.setindex!(FEB::FEVectorBlock, v, ::Colon) = (FEB.entries[FEB.offset+1:FEB.last_index] = v)
Base.setindex!(FEB::FEVectorBlock, v, i::AbstractArray) = (FEB.entries[FEB.offset.+i] = v)
Base.size(FEF::FEVector)=size(FEF.FEVectorBlocks)
Base.size(FEB::FEVectorBlock)=FEB.last_index-FEB.offset
Base.length(FEF::FEVector)=length(FEF.FEVectorBlocks)
Base.length(FEB::FEVectorBlock)=FEB.last_index-FEB.offset


function Base.append!(FEF::FEVector{T},name::String,FES::FESpace) where T <: Real
    append!(FEF.entries,zeros(T,FES.ndofs))
    newBlock = FEVectorBlock{T}(name, FES, FEF.FEVectorBlocks[end].last_index , FEF.FEVectorBlocks[end].last_index+FES.ndofs, FEF.entries)
    push!(FEF.FEVectorBlocks,newBlock)
end

################
### FEMatrix ###
################
#
# used to store (sparse) matrix representations of PDEOperators for FESpaces
# and can have several blocks of different FESpaces
# acts like an AbstractArray{T,2}

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

struct FEMatrix{T} <: AbstractArray{T,1}
    FEMatrixBlocks::Array{FEMatrixBlock{T},1}
    nFE::Int
    entries::AbstractMatrix
end

# show function for FEMatrix
function Base.show(io::IO, FEM::FEMatrix) where {T}
	println("\nFEMatrix information")
    println("====================")
    println("  block |  ndofsX |  ndofsY | name (FETypeX, FETypeY) ")
    for j=1:length(FEM)
        n = mod(j-1,FEM.nFE)+1
        m = Int(floor(j/FEM.nFE))
        @printf("  [%d,%d] |",m,n);
        @printf("  %6d |",FEM[j].FESX.ndofs);
        @printf("  %6d |",FEM[j].FESY.ndofs);
        @printf(" %s (%s,%s)\n",FEM[j].name,FEM[j].FESX.name,FEM[j].FESY.name);
    end    
end

function FEMatrix{T}(name::String, FES::FESpace) where T <: Real
    entries = ExtendableSparseMatrix{T,Int32}(FES.ndofs,FES.ndofs)
    Block = FEMatrixBlock{T}(name, FES, FES, 0 , 0, FES.ndofs, FES.ndofs, entries)
    return FEMatrix{T}([Block], 1, entries)
end

function FEMatrix{T}(name::String, FESX::FESpace, FESY::FESpace) where T <: Real
    entries = ExtendableSparseMatrix{T,Int32}(FETypeX.ndofs,FESY.ndofs)
    Block = FEMatrixBlock{T}(name, FESX, FESY, 0 , 0, FESX.ndofs, FETypeY.ndofs, entries)
    return FEMatrix{T}([Block], 2, entries)
end

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

function Base.fill!(B::FEMatrixBlock, value::Real)
    row = 0
    for col = B.offsetY+1:B.last_indexY
        for r in nzrange(B.entries.cscmatrix, col)
            row = rowvals(B.entries.cscmatrix)[r]
            if row >= B.offsetX && row <= B.last_indexX
                B.entries.cscmatrix.nzval[r] = value
            end
        end
    end
end




Base.getindex(FEF::FEMatrix,i) = FEF.FEMatrixBlocks[i]
Base.getindex(FEF::FEMatrix,i,j) = FEF.FEMatrixBlocks[(i-1)*FEF.nFE+j]
Base.getindex(FEB::FEMatrixBlock,i::Int,j::Int)=FEB.entries[FEB.offsetX+i,FEB.offsetY+j]
Base.getindex(FEB::FEMatrixBlock,i::Any,j::Any)=FEB.entries[FEB.offsetX.+i,FEB.offsetY.+j]
Base.setindex!(FEB::FEMatrixBlock, v, i::Int, j::Int) = (FEB.entries[FEB.offsetX+i,FEB.offsetY+j] = v)
Base.size(FEF::FEMatrix)=length(FEF.FEMatrixBlocks)
Base.size(FEB::FEMatrixBlock)=[FEB.last_indexX-FEB.offsetX,FEB.last_indexY-FEB.offsetY]
