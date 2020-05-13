
################
### FEVector ###
################

struct FEVectorBlock{T} <: AbstractArray{T,1}
    name::String
    FEType::AbstractFiniteElement
    offset::Int
    last_index::Int
    entries::Array{T,1} # shares with parent object
end

struct FEVector{T} <: AbstractArray{T,1}
    FEVectorBlocks::Array{FEVectorBlock{T},1}
    entries::Array{T,1}
end

# show function for FEVector
function show(FEF::FEVector)
	println("\nFEVector information")
    println("=========================")
    println("   nblocks = $(length(FEF))")
    for j=1:length(FEF.FEVectorBlocks)
        println("  block[$j] = $(FEF[j].name) (FEtype = $(FEF[j].FEType.name), ndof = $(FEF[j].FEType.ndofs))");
    end    
end

function FEVector{T}(name::String, FEType::AbstractFiniteElement) where T <: Real
    entries = zeros(T,FEType.ndofs)
    Block = FEVectorBlock{T}(name, FEType, 0 , size(entries,1), entries)
    return FEVector{T}([Block], entries)
end

function FEVector{T}(name::String, FETypes::Array{<:AbstractFiniteElement,1}) where T <: Real
    ndofs = 0
    for j = 1:length(FETypes)
        ndofs += FETypes[j].ndofs
    end    
    entries = zeros(T,ndofs)
    Blocks = Array{FEVectorBlock,1}(undef,length(FETypes))
    offset = 0
    for j = 1:length(FETypes)
        Blocks[j] = FEVectorBlock{T}(name, FETypes[j], offset , offset+FETypes[j].ndofs, entries)
        offset += FETypes[j].ndofs
    end    
    return FEVector{T}(Blocks, entries)
end


Base.getindex(FEF::FEVector,i) = FEF.FEVectorBlocks[i]
Base.getindex(FEB::FEVectorBlock,i::Int)=FEB.entries[FEB.offset+i]
Base.getindex(FEB::FEVectorBlock,i::AbstractArray)=FEB.entries[FEB.offset.+i]
Base.getindex(FEB::FEVectorBlock,::Colon)=FEB.entries[FEB.offset+1:FEB.last_index]
Base.setindex!(FEB::FEVectorBlock, v, i::Int) = (FEB.entries[FEB.offset+i] = v)
Base.setindex!(FEB::FEVectorBlock, v, ::Colon) = (FEB.entries[FEB.offset+1:FEB.last_index] = v)
Base.setindex!(FEB::FEVectorBlock, v, i::AbstractArray) = (FEB.entries[FEB.offset.+i] = v)
Base.length(FEF::FEVector)=length(FEF.FEVectorBlocks)
Base.length(FEB::FEVectorBlock)=FEB.last_index-FEB.offset


function Base.append!(FEF::FEVector{T},name::String,FEType::AbstractFiniteElement) where T <: Real
    append!(FEF.entries,zeros(T,FEType.ndofs))
    newBlock = FEVectorBlock{T}(name, FEType, FEF.FEVectorBlocks[end].last_index , FEF.FEVectorBlocks[end].last_index+FEType.ndofs, FEF.entries)
    push!(FEF.FEVectorBlocks,newBlock)
end

################
### FEMatrix ###
################

struct FEMatrixBlock{T} <: AbstractArray{T,2}
    name::String
    FETypeX::AbstractFiniteElement
    FETypeY::AbstractFiniteElement
    offsetX::Int32
    offsetY::Int32
    last_indexX::Int32
    last_indexY::Int32
    entries::AbstractMatrix # shares with parent object
end

struct FEMatrix{T} <: AbstractArray{T,1}
    FEMatrixBlocks::Array{FEMatrixBlock{T},1}
    entries::AbstractMatrix
end

# show function for FEMatrix
function show(FEM::FEMatrix)
	println("\nFEMatrix information")
    println("=========================")
    println("   nblocks = $(length(FEM))")
    for j=1:length(FEF.FEVectorBlocks)
        println("  block[$j] = $(FEM[j].name) (FEtypeX/Y = $(FEM[j].FETypeX.name)/$(FEM[j].FETypeX.name), size = $(FEM[j].FETypeX.ndofs)x$(FEM[j].FETypeY.ndofs))");
    end    
end

function FEMatrix{T}(name::String, FEType::AbstractFiniteElement) where T <: Real
    entries = ExtendableSparseMatrix{T,Int32}(FEType.ndofs,FEType.ndofs)
    Block = FEMatrixBlock{T}(name, FEType, FEType, 0 , 0, FEType.ndofs, FEType.ndofs, entries)
    return FEMatrix{T}([Block], entries)
end

function FEMatrix{T}(name::String, FETypeX::AbstractFiniteElement, FETypeY::AbstractFiniteElement) where T <: Real
    entries = ExtendableSparseMatrix{T,Int32}(FETypeX.ndofs,FETypeY.ndofs)
    Block = FEMatrixBlock{T}(name, FETypeX, FETypeY, 0 , 0, FETypeX.ndofs, FETypeY.ndofs, entries)
    return FEMatrix{T}([Block], entries)
end

function FEMatrix{T}(name::String, FETypes::Array{<:AbstractFiniteElement,1}) where T <: Real
    ndofs = 0
    for j=1:length(FETypes)
        ndofs += FETypes[j].ndofs
    end
    entries = ExtendableSparseMatrix{T,Int32}(ndofs,ndofs)

    Blocks = Array{FEMatrixBlock{T},1}(undef,length(FETypes)^2)
    offsetX = 0
    offsetY = 0
    for j=1:length(FETypes)
        offsetY = 0
        for k=1:length(FETypes)
            Blocks[(j-1)*length(FETypes)+k] = FEMatrixBlock{T}(name * " [$j,$k]", FETypes[j], FETypes[k], offsetX , offsetY, offsetX+FETypes[j].ndofs, offsetY+FETypes[k].ndofs, entries)
            offsetY += FETypes[k].ndofs
        end    
        offsetX += FETypes[j].ndofs
    end    
    
    return FEMatrix{T}(Blocks, entries)
end





Base.getindex(FEF::FEMatrix,i) = FEF.FEMatrixBlocks[i]
Base.getindex(FEF::FEMatrix,i,j) = FEF.FEMatrixBlocks[(i-1)*length(FEF.FEMatrixBlocks)+j]
Base.getindex(FEB::FEMatrixBlock,i::Int,j::Int)=FEB.entries[FEB.offsetX+i,FEB.offsetY+j]
Base.getindex(FEB::FEMatrixBlock,i::Any,j::Any)=FEB.entries[FEB.offsetX.+i,FEB.offsetY.+j]
Base.setindex!(FEB::FEMatrixBlock, v, i::Int, j::Int) = (FEB.entries[FEB.offsetX+i,FEB.offsetY+j] = v)
Base.size(FEF::FEMatrix)=length(FEF.FEMatrixBlocks)
Base.size(FEB::FEMatrixBlock)=[FEB.last_indexX-FEB.offsetX,FEB.last_indexY-FEB.offsetY]
