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
struct FEVectorBlock{T,Tv,Ti,FEType,APT} <: AbstractArray{T,1}
    name::String
    FES::FESpace{Tv,Ti,FEType,APT}
    offset::Int
    last_index::Int
    entries::Array{T,1} # shares with parent object
end

"""
$(TYPEDEF)

a plain array but with an additional layer of several FEVectorBlock subdivisions each carrying coefficients for their associated FESpace
"""
struct FEVector{T,Tv,Ti}
    FEVectorBlocks::Array{FEVectorBlock{T,Tv,Ti},1}
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

function LinearAlgebra.norm(FEV::FEVector{T}, p::Real = 2) where {T}
    norms = zeros(T,length(FEV))
    for j = 1 : length(FEV)
        norms[j] = 0
        # norms[j] = norm(FEV[j], p) # does not work, why???
        for k = 1 : length(FEV[j])
            norms[j] += FEV.entries[FEV[j].offset + k]^p
        end
        norms[j] = norms[j]^(1/p)
    end
    return norms
end

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
function FEVector(name::String, FES::FESpace{Tv,Ti,FEType,APT}) where {Tv,Ti,FEType,APT}
    return FEVector{Float64}([name],[FES])
end
function FEVector{T}(name::String, FES::FESpace{Tv,Ti,FEType,APT}) where {T,Tv,Ti,FEType,APT}
    return FEVector{T}([name],[FES])
end

"""
````
FEVector{T}(name::String, FES::Array{FESpace,1}) where T <: Real
````

Creates FEVector that has one block for each FESpace in FES.
"""
function FEVector(name, FES::Array{<:FESpace{Tv,Ti},1}) where {Tv,Ti}
    return FEVector{Float64}(name,FES)
end
function FEVector{T}(name::Array{String,1}, FES::Array{<:FESpace{Tv,Ti},1}) where {T,Tv,Ti}
    @logmsg DeepInfo "Creating FEVector mit blocks $((p->p.name).(FES))"
    ndofs = 0
    for j = 1:length(FES)
        ndofs += FES[j].ndofs
    end    
    entries = zeros(T,ndofs)
    Blocks = Array{FEVectorBlock{T,Tv,Ti},1}(undef,length(FES))
    offset = 0
    for j = 1:length(FES)
        Blocks[j] = FEVectorBlock{T,Tv,Ti,eltype(FES[j]),assemblytype(FES[j])}(name[j], FES[j], offset , offset+FES[j].ndofs, entries)
        offset += FES[j].ndofs
    end    
    return FEVector{T,Tv,Ti}(Blocks, entries)
end
function FEVector{T}(name::String, FES::Array{<:FESpace{Tv,Ti},1}) where {T,Tv,Ti}
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
	println(io,"\nFEVector information")
    println(io,"====================")
    println(io,"   block  |  ndofs  | name (FEType) ")
    for j=1:length(FEF)
        @printf(io," [%5d]  | ",j);
        @printf(io," %6d |",FEF[j].FES.ndofs);
        @printf(io," %s (%s)\n",FEF[j].name,FEF[j].FES.name);
    end    
end



"""
$(TYPEDSIGNATURES)

Custom `append` function for `FEVector` that adds a FEVectorBlock at the end.
"""
function Base.append!(FEF::FEVector{T},name::String,FES::FESpace{Tv,Ti,FEType,APT}) where {T,Tv,Ti,FEType,APT}
    append!(FEF.entries,zeros(T,FES.ndofs))
    newBlock = FEVectorBlock{T,Tv,Ti,FEType,APT}(name, FES, FEF.FEVectorBlocks[end].last_index , FEF.FEVectorBlocks[end].last_index+FES.ndofs, FEF.entries)
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
    l::Int = length(b)
    aoffset::Int = a.offset
    boffset::Int = b.offset
    for j = 1 : l
        a.entries[aoffset+j] += b.entries[boffset+j] * factor
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Adds Array b to FEVectorBlock a.
"""
function addblock!(a::FEVectorBlock, b::AbstractVector; factor = 1)
    aoffset::Int = a.offset
    for j = 1 : length(b)
        a.entries[aoffset+j] += b[j] * factor
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Scalar product between two FEVEctorBlocks
"""
function LinearAlgebra.dot(a::FEVectorBlock{T}, b::FEVectorBlock{T}) where {T}
    aoffset::Int = a.offset
    boffset::Int = b.offset
    result::T = 0.0
    for j = 1 : length(b)
       result += a.entries[aoffset+j] * b.entries[boffset+j]
    end
    return result
end
