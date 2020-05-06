struct FEVectorBlock{T} <: AbstractArray{T,1}
    name::String
    FEType::AbstractFiniteElement
    offset::Int
    last_index::Int
    coefficients::Array{T,1} # shares with parent object
end

struct FEVector{T} <: AbstractArray{T,1}
    FEVectorBlocks::Array{FEVectorBlock{T},1}
    coefficients::Array{T,1}
end

# show function for FiniteElement
function show(FEF::FEVector)
	println("\nFEVector information")
    println("=========================")
    println("   nblocks = $(length(FEF))")
    for j=1:length(FEF.FEVectorBlocks)
        println("  block[$j] = $(FEF[j].name) (FEtype = $(FEF[j].FEType.name), ndof = $(FEF[j].FEType.ndofs))");
    end    
end

function FEVector{T}(name::String, FEType::AbstractFiniteElement) where T <: Real
    coefficients = zeros(T,FEType.ndofs)
    Block = FEVectorBlock{T}(name, FEType, 0 , size(coefficients,1), coefficients)
    return FEVector{T}([Block], coefficients)
end


Base.getindex(FEF::FEVector,i) = FEF.FEVectorBlocks[i]
Base.getindex(FEB::FEVectorBlock,i::Int)=FEB.coefficients[FEB.offset+i]
Base.getindex(FEB::FEVectorBlock,i::AbstractArray)=FEB.coefficients[FEB.offset.+i]
Base.getindex(FEB::FEVectorBlock,::Colon)=FEB.coefficients[FEB.offset+1:FEB.last_index]
Base.setindex!(FEB::FEVectorBlock, v, i::Int) = (FEB.coefficients[FEB.offset+i] = v)
Base.setindex!(FEB::FEVectorBlock, v, ::Colon) = (FEB.coefficients[FEB.offset+1:FEB.last_index] = v)
Base.setindex!(FEB::FEVectorBlock, v, i::AbstractArray) = (FEB.coefficients[FEB.offset.+i] = v)
Base.length(FEF::FEVector)=length(FEF.FEVectorBlocks)
Base.length(FEB::FEVectorBlock)=FEB.last_index-FEB.offset


function Base.append!(FEF::FEVector{T},name::String,FEType::AbstractFiniteElement) where T <: Real
    append!(FEF.coefficients,zeros(T,FEType.ndofs))
    newBlock = FEVectorBlock{T}(name, FEType, FEF.FEVectorBlocks[end].last_index , FEF.FEVectorBlocks[end].last_index+FEType.ndofs, coefficients)
    push!(FEF.FEVectorBlocks,newBlock)
end
