###################
# QuadratureRules #
###################
#
# here all quadrature rules for the different ElementGeometries are collected
# there are some hard-coded ones for the lowest-order rules (that might be extended later)
# and also generic functions that generate rules of arbitrary order
#
# integrate! allows to intgrate cell-wise (order face-wise etc. depending on the AbstractAssemblyType)
# integrate does the same but only returns the full integral and is more memory-friendly

"""
$(TYPEDEF)

A struct that contains the name of the quadrature rule, the reference points and the weights for the parameter-determined element geometry.
"""
struct QuadratureRule{T <: Real, ET <: AbstractElementGeometry}
    name::String
    xref::Array{Array{T, 1}}
    w::Array{T, 1}
end

"""
$(TYPEDSIGNATURES)

Custom `eltype` function for `QuadratureRule{T,ET}`.
"""
Base.eltype(::QuadratureRule{T,ET}) where{T <: Real, ET <: AbstractElementGeometry} = [T,ET]

"""
$(TYPEDSIGNATURES)

Custom `show` function for `QuadratureRule{T,ET}` that prints some information.
"""
function Base.show(io::IO, Q::QuadratureRule{T,ET} where{T <: Real, ET <: AbstractElementGeometry})
    npoints = length(Q.xref);
    println("QuadratureRule information");
    println("    shape ; $(eltype(Q)[2])")
	println("     name : $(Q.name)");
	println("  npoints : $(npoints) ($(eltype(Q)[1]))")
end

# sets up a quadrature rule that evaluates at vertices of element geometry
# not optimal from quadrature point of view, but helpful when defining nodal interpolations
function VertexRule(ET::Type{Edge1D})
    xref = [[0],[1]]
    w = [1//2, 1//2]
    return QuadratureRule{Float64, ET}("vertex rule edge", xref, w)
end
function VertexRule(ET::Type{Triangle2D})
    xref = [[0, 0], [1,0], [0,1]]
    w = [1//3, 1//3, 1//3]
    return QuadratureRule{Float64, ET}("vertex rule triangle", xref, w)
end
function VertexRule(ET::Type{Parallelogram2D})
    xref = [[0, 0], [1,0], [1,1], [0,1]]
    w = [1//4, 1//4, 1//4, 1//4]
    return QuadratureRule{Float64, ET}("vertex rule parallelogram", xref, w)
end

"""
````
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: AbstractElementGeometry1D}
````

Constructs 1D quadrature rule of specified order.
"""
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: AbstractElementGeometry1D}
    if order <= 1
        name = "midpoint rule"
        xref = Vector{Array{T,1}}(undef,1);
        xref[1] = ones(T,1) * 1 // 2
        w = [1]
    elseif order == 2
        name = "Simpson's rule"
        xref = Vector{Array{T,1}}(undef,3);
        xref[1] = [0];
        xref[2] = [1//2];
        xref[3] = [1];
        w = [1//6; 2//3; 1//6]     
    else
        name = "generic Gauss rule of order $order"
        xref, w = get_generic_quadrature_Gauss(order)
    end
    return QuadratureRule{T, ET}(name, xref, w)
end

"""
````
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: AbstractElementGeometry0D}
````

Constructs 0D quadrature rule of specified order (always point evaluation).
"""
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: AbstractElementGeometry0D}
    name = "point evaluation"
    xref = Vector{Array{T,1}}(undef,1);
    xref[1] = ones(T,1)
    w = [1]
    return QuadratureRule{T, ET}(name, xref, w)
end


"""
````
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Triangle2D}
````

Constructs quadrature rule on Triangle2D of specified order.
"""
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Triangle2D}
  if order <= 1
      name = "midpoint rule"
      xref = Vector{Array{T,1}}(undef,1);
      xref[1] = ones(T,2) * 1 // 3
      w = [1]
  elseif order == 2 # face midpoint rule  
      name = "face midpoints rule"
      xref = Vector{Array{T,1}}(undef,3);
      xref[1] = [1//2,1//2];
      xref[2] = [0//1,1//2];
      xref[3] = [1//2,0//1];
      w = [1//3; 1//3; 1//3]     
  else
      name = "generic Stroud rule of order $order"
      xref, w = get_generic_quadrature_Stroud(order)
  end
  return QuadratureRule{T, ET}(name, xref, w)
end


"""
````
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Parallelogram2D}
````

Constructs quadrature rule on Parallelogram2D of specified order.
"""
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Parallelogram2D}
  if order <= 1
      name = "midpoint rule"
      xref = Vector{Array{T,1}}(undef,1);
      xref[1] = ones(T,2) * 1 // 2
      w = [1]
  else
      name = "generic Gauss tensor rule of order $order"
      xref1D, w1D = get_generic_quadrature_Gauss(order)
      xref = Vector{Array{T,1}}(undef,length(xref1D)^2)
      w = zeros(T,length(xref1D)^2)
      index = 1
      for j = 1 : length(xref1D), k = 1 : length(xref1D)
        xref[index] = zeros(T,2)
        xref[index][1] = xref1D[j][1]
        xref[index][2] = xref1D[k][1]
        w[index] = w1D[j] * w1D[k]
        index += 1
      end
  end
  return QuadratureRule{T, ET}(name, xref, w)
end


"""
````
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Parallelepiped3D}
````

Constructs quadrature rule on Parallelepiped3D of specified order.
"""
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Parallelepiped3D}
  if order <= 1
      name = "midpoint rule"
      xref = Vector{Array{T,1}}(undef,1);
      xref[1] = ones(T,3) * 1 // 2
      w = [1]
  else
      name = "generic Gauss tensor rule of order $order"
      xref1D, w1D = get_generic_quadrature_Gauss(order)
      xref = Vector{Array{T,1}}(undef,length(xref1D)^3)
      w = zeros(T,length(xref1D)^3)
      index = 1
      for j = 1 : length(xref1D), k = 1 : length(xref1D), l = 1 : length(xref1D)
        xref[index] = zeros(T,3)
        xref[index][1] = xref1D[j][1]
        xref[index][2] = xref1D[k][1]
        xref[index][3] = xref1D[l][1]
        w[index] = w1D[j] * w1D[k] * w1D[l]
        index += 1
      end
  end
  return QuadratureRule{T, ET}(name, xref, w)
end


"""
````
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Tetrahedron3D}
````

Constructs quadrature rule on Tetrahedron3D of specified order.
"""
function QuadratureRule{T,ET}(order::Int) where {T<:Real, ET <: Tetrahedron3D}
  if order <= 1
      name = "midpoint rule"
      xref = Vector{Array{T,1}}(undef,1);
      xref[1] = ones(T,3) * 1 // 4
      w = [1]
  elseif order > 1
      name = "order 2 rule"
      xref = Vector{Array{T,1}}(undef,4);
      xref[1] = [0.1381966011250105,0.1381966011250105,0.1381966011250105]
      xref[2] = [0.5854101966249685,0.1381966011250105,0.1381966011250105]
      xref[3] = [0.1381966011250105,0.5854101966249685,0.1381966011250105]
      xref[4] = [0.1381966011250105,0.1381966011250105,0.5854101966249685]
      w = ones(T,4) * 1 // 4
  else
      # no generic rule implemented yet
  end
  return QuadratureRule{T, ET}(name, xref, w)
end


function get_generic_quadrature_Gauss(order::Int)
    ngpts::Int = div(order, 2) + 1
    
    # compute 1D Gauss points on interval [-1,1] and weights
    gamma = (1 : ngpts-1) ./ sqrt.(4 .* (1 : ngpts-1).^2 .- ones(ngpts-1,1) );
    F = eigen(diagm(1 => gamma[:], -1 => gamma[:]));
    r = F.values;
    w = 2*F.vectors[1,:].^2;
    
    # transform to interval [0,1]
    r = .5 .* r .+ .5;
    w = .5 .* w';

    xref = Array{Array{Float64,1}}(undef,length(r))
    for j = 1 : length(r)
        xref[j] = [r[j]];
    end
    
    return xref, w[:]
end
  
# computes quadrature points and weights by Stroud Conical Product rule
function get_generic_quadrature_Stroud(order::Int)
    ngpts::Int = div(order, 2) + 1
    
    # compute 1D Gauss points on interval [-1,1] and weights
    gamma = (1 : ngpts-1) ./ sqrt.(4 .* (1 : ngpts-1).^2 .- ones(ngpts-1,1) );
    F = eigen(diagm(1 => gamma[:], -1 => gamma[:]));
    r = F.values;
    a = 2*F.vectors[1,:].^2;
    
    # compute 1D Gauss-Jacobi Points for Intervall [-1,1] and weights
    delta = -1 ./ (4 .* (1 : ngpts).^2 .- ones(ngpts,1));
    gamma = sqrt.((2 : ngpts) .* (1 : ngpts-1)) ./ (2 .* (2 : ngpts) .- ones(ngpts-1,1));
    F = eigen(diagm(0 => delta[:], 1 => gamma[:], -1 => gamma[:]));
    s = F.values;
    b = 2*F.vectors[1,:].^2;
    
    # transform to interval [0,1]
    r = .5 .* r .+ .5;
    s = .5 .* s .+ .5;
    a = .5 .* a';
    b = .5 .* b';
    
    # apply conical product rule
    # xref[:,[1 2]] = [ s_j , r_i(1-s_j) ] 
    # w = a_i*b_j
    s = repeat(s',ngpts,1)[:];
    r = repeat(r,ngpts,1);
    xref = Array{Array{Float64,1}}(undef,length(s))
    for j = 1 : length(s)
        xref[j] = s[j].*[1,0] - r[j]*(s[j]-1).*[0,1];
    end
    w = a'*b;
    
    return xref, w[:]
end


"""
$(TYPEDSIGNATURES)

Integration that writes result on every item into integral4items.
"""
function integrate!(
    integral4items::AbstractArray,
    grid::ExtendableGrid,
    AT::Type{<:AbstractAssemblyType},
    integrand!::Function,
    order::Int,
    resultdim::Int;
    verbosity::Int = 0,
    index_offset::Int = 0,
    item_dependent_integrand::Bool = false)
    
    xCoords = grid[Coordinates]
    dim = size(xCoords,1)
    xItemNodes = grid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = grid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = grid[GridComponentGeometries4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    NumberType = eltype(integral4items)
    
    # find proper quadrature rules
    EG = Base.unique(xItemGeometries)
    qf = Array{QuadratureRule,1}(undef,length(EG))
    local2global = Array{L2GTransformer,1}(undef,length(EG))
    for j = 1 : length(EG)
        qf[j] = QuadratureRule{NumberType,EG[j]}(order);
        local2global[j] = L2GTransformer{NumberType,EG[j],grid[CoordinateSystem]}(grid,AT)
    end    
    if verbosity > 0
        println("INTEGRATE")
        println("=========")
        println("nitems = $nitems")
        for j = 1 : length(EG)
            println("QuadratureRule [$j] for $(EG[j]):")
            show(qf[j])
        end
    end

    item_integrand!(result,x,item,xref) = item_dependent_integrand ? integrand!(result,x,item,xref) : integrand!(result,x)

    # loop over items
    x = zeros(NumberType, dim)
    result = zeros(NumberType, resultdim)
    itemET = xItemGeometries[1]
    iEG = 1
    if resultdim == 1
        for item = 1 : nitems
            integral4items[item+index_offset] = 0

            # find index for CellType
            itemET = xItemGeometries[item]
            iEG = findfirst(isequal(itemET), EG)

            update!(local2global[iEG],item)

            for i in eachindex(qf[iEG].w)
                eval!(x, local2global[iEG], qf[iEG].xref[i])
                item_integrand!(result,x,item, qf[iEG].xref[i])
                integral4items[item+index_offset] += result[1] * qf[iEG].w[i] * xItemVolumes[item];
            end  
        end
    else
        for item = 1 : nitems
            integral4items[item] = 0

            # find index for CellType
            itemET = xItemGeometries[item]
            iEG = findfirst(isequal(itemET), EG)

            update!(local2global[iEG],item)

            for i in eachindex(qf[iEG].w)
                eval!(x, local2global[iEG], qf[iEG].xref[i])
                item_integrand!(result,x,item,qf[iEG].xref[i])
                for j = 1 : resultdim
                    integral4items[item, j] += result[j] * qf[iEG].w[i] * xItemVolumes[item];
                end
            end  
        end
    end
end

"""
$(TYPEDSIGNATURES)

Integration that returns total integral.
"""
function integrate(grid::ExtendableGrid, AT::Type{<:AbstractAssemblyType}, integrand!::Function, order::Int, resultdim::Int; verbosity::Int = 0)
    xCoords = grid[Coordinates]
    dim = size(xCoords,1)
    xItemNodes = grid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes = grid[GridComponentVolumes4AssemblyType(AT)]
    xItemGeometries = grid[GridComponentGeometries4AssemblyType(AT)]
    nitems = num_sources(xItemNodes)
    NumberType = eltype(xCoords)
    
    # find proper quadrature rules
    EG = Base.unique(xItemGeometries)
    qf = Array{QuadratureRule,1}(undef,length(EG))
    local2global = Array{L2GTransformer,1}(undef,length(EG))
    for j = 1 : length(EG)
        qf[j] = QuadratureRule{NumberType,EG[j]}(order);
        local2global[j] = L2GTransformer{NumberType,EG[j],grid[CoordinateSystem]}(grid,AT)
    end    
    if verbosity > 0
        println("INTEGRATE")
        println("=========")
        println("nitems = $nitems")
        for j = 1 : length(EG)
            println("QuadratureRule [$j] for $(EG[j]):")
            show(qf[j])
        end
    end

    # loop over items
    x = zeros(NumberType, dim)
    result = zeros(NumberType, resultdim)
    itemET = xItemGeometries[1]
    iEG = 1
    integral = zeros(NumberType, resultdim)
    for item = 1 : nitems
        # find index for CellType
        itemET = xItemGeometries[item]
        iEG = findfirst(isequal(itemET), EG)

        update!(local2global[iEG],item)

        for i in eachindex(qf[iEG].w)
            eval!(x, local2global[iEG], qf[iEG].xref[i])
            integrand!(result,x)
            for j = 1 : resultdim
                integral[j] += result[j] * qf[iEG].w[i] * xItemVolumes[item];
            end
        end  
    end

    return integral
end