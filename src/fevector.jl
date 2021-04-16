################
### FEVector ###
################
#
# used to store coefficients for FESpaces and can have several blocks of different FESpaces
# acts like an AbstractArray{T,1}

"""
$(TYPEDEF)

block of an FEVector that carries coefficients for an associated FESpace and can be assigned as an AbstractArray (getindex, setindex, size, length)
"""
struct FEVectorBlock{T} <: AbstractArray{T,1}
    name::String
    FES::FESpace
    offset::Int
    last_index::Int
    entries::Array{T,1} # shares with parent object
end

"""
$(TYPEDEF)

a plain array but with an additional layer of several FEVectorBlock subdivisions each carrying coefficients for their associated FESpace
"""
struct FEVector{T} <: AbstractArray{T,1}
    FEVectorBlocks::Array{FEVectorBlock{T},1}
    entries::Array{T,1}
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

"""
$(TYPEDSIGNATURES)

Custom `length` function for `FEVector` that gives the number of defined FEMatrixBlocks in it
"""
Base.length(FEF::FEVector)=length(FEF.FEVectorBlocks)

"""
$(TYPEDSIGNATURES)

Custom `length` function for `FEVectorBlock` that gives the coressponding number of degrees of freedoms of the associated FESpace
"""
Base.length(FEB::FEVectorBlock)=FEB.last_index-FEB.offset

"""
````
FEVector{T}(name::String, FES::FESpace) where T <: Real
````

Creates FEVector that has one block.
"""
function FEVector{T}(name::String, FES::FESpace) where T <: Real
    return FEVector{T}([name],[FES])
end

"""
````
FEVector{T}(name::String, FES::Array{FESpace,1}) where T <: Real
````

Creates FEVector that has one block for each FESpace in FES.
"""
function FEVector{T}(name::Array{String,1}, FES::Array{<:FESpace,1}) where T <: Real
    @logmsg DeepInfo "Creating FEVector mit blocks $((p->p.name).(FES))"
    ndofs = 0
    for j = 1:length(FES)
        ndofs += FES[j].ndofs
    end    
    entries = zeros(T,ndofs)
    Blocks = Array{FEVectorBlock,1}(undef,length(FES))
    offset = 0
    for j = 1:length(FES)
        Blocks[j] = FEVectorBlock{T}(name[j], FES[j], offset , offset+FES[j].ndofs, entries)
        offset += FES[j].ndofs
    end    
    return FEVector{T}(Blocks, entries)
end
function FEVector{T}(name::String, FES::Array{<:FESpace,1}) where T <: Real
    names = Array{String,1}(undef, length(FES))
    for j = 1:length(FES)
        names[j] = name * " [$j]"
    end    
    FEVector{T}(names, FES)
end


"""
$(TYPEDSIGNATURES)

Custom `show` function for `FEVector` that prints some information on its blocks.
"""
function Base.show(io::IO, FEF::FEVector)
	println("\nFEVector information")
    println("====================")
    println("   block  |  ndofs  | name (FEType) ")
    for j=1:length(FEF)
        @printf(" [%5d]  | ",j);
        @printf(" %6d |",FEF[j].FES.ndofs);
        @printf(" %s (%s)\n",FEF[j].name,FEF[j].FES.name);
    end    
end



"""
$(TYPEDSIGNATURES)

Custom `append` function for `FEVector` that adds a FEVectorBlock at the end.
"""
function Base.append!(FEF::FEVector{T},name::String,FES::FESpace) where T <: Real
    append!(FEF.entries,zeros(T,FES.ndofs))
    newBlock = FEVectorBlock{T}(name, FES, FEF.FEVectorBlocks[end].last_index , FEF.FEVectorBlocks[end].last_index+FES.ndofs, FEF.entries)
    push!(FEF.FEVectorBlocks,newBlock)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Custom `fill` function for `FEVectorBlock` (only fills the block, not the complete FEVector).
"""
function Base.fill!(b::FEVectorBlock, value)
    for j = b.offset+1 : b.last_index
        b.entries[j] = value
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Adds FEVectorBlock b to FEVectorBlock a.
"""
function addblock!(a::FEVectorBlock, b::FEVectorBlock; factor = 1)
    for j = b.offset+1 : b.last_index
        a.entries[a.offset+j] += b.entries[j] * factor
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Adds Array b to FEVectorBlock a.
"""
function addblock!(a::FEVectorBlock, b::AbstractVector; factor = 1)
    for j = 1 : length(b)
        a.entries[a.offset+j] += b[j] * factor
    end
    return nothing
end
