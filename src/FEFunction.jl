struct FEFunction{T} <: AbstractVector{T}
    blocknames::Array{String,1}
    FETypes::Array{AbstractFiniteElement,1}
    offsets::Vector{Int}
    coefficients::Vector{T}
end


# show function for FiniteElement
function show(FEF::FEFunction)
	println("\nFiniteElementFunction information")
    println("=================================")
    for j=1:length(FEF.blocknames)
        println("   [$j] $(FEF.blocknames[j]) (FEtype = $(FEF.FETypes[j].name), ndof = $(FEF.FETypes[j].ndofs))");
    end    
end

function FEFunction{T}(name::String, FEType::AbstractFiniteElement) where T <: Real
    return FEFunction{T}([name], [FEType], [0, FEType.ndofs], zeros(T,FEType.ndofs))
end

function FEFunction{T}(names::Array{String,1}, FETypes::Array{AbstractFiniteElement,1}) where T <: Real
    offsets = zeros(Int32,length(FETypes)+1)
    for j = 2 : length(FETypes)+1
        offsets[j] = offsets[j-1] + FETypes[j-1].ndofs
    end    
    return FEFunction{T}(names, FETypes, offsets, zeros(T,offsets[end]))
end

Base.getindex(FEF::FEFunction,i)=FEF.coefficients[i]
Base.getindex(FEF::FEFunction,::Colon)=FEF.coefficients[:]
Base.getindex(FEF::FEFunction,i::Int,j)=FEF.coefficients[FEF.offsets[i]+j]
Base.getindex(FEF::FEFunction,i::Int,::Colon)=FEF.coefficients[FEF.offsets[i]+1:FEF.offsets[i+1]]
Base.setindex!(FEF::FEFunction, v, i) = (FEF.coefficients[i] = v)
Base.setindex!(FEF::FEFunction, v, i::Int, j) = (FEF.coefficients[FEF.offsets[i]+j] = v)
Base.setindex!(FEF::FEFunction, v, i::Int, ::Colon) = (FEF.coefficients[FEF.offsets[i]+1:FEF.offsets[i+1]] = v)
Base.size(FEF::FEFunction)=[length(FEF.FETypes),FEF.offsets[end]]

function Base.append!(FEF::FEFunction{T},name::String,FEType::AbstractFiniteElement) where T <: Real
    push!(FEF.blocknames,name)
    push!(FEF.FETypes,FEType)
    push!(FEF.offsets,FEF.offsets[end]+FEType.ndofs)
    append!(FEF.coefficients,zeros(T,FEType.ndofs))
end
