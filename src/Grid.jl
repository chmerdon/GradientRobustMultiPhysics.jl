module Grid

using SparseArrays
using LinearAlgebra

export Mesh, ensure_volume4cells!, ensure_bfaces!, ensure_faces4cells!, ensure_nodes4faces!, ensure_cells4faces!, ensure_normal4faces!, ensure_length4faces!, ensure_signs4cells!, get_boundary_grid

abstract type AbstractElemType end

mutable struct Mesh{T <: Real}
    coords4nodes::Array{T,2}
    nodes4cells::Array{Int64,2}

    elemtypes::Array{AbstractElemType,1}
    volume4cells::Array{T,1}
    nodes4faces::Array{Int64,2}
    faces4cells::Array{Int64,2}
    bfaces::Array{Int64,1}
    bregions::Array{Int64,1}
    cells4faces::Array{Int64,2}
    length4faces::Array{T,1}
    normal4faces::Array{T,2}
    signs4cells::Array{Int64,2}
    
    function Mesh{T}(coords,nodes,ET::AbstractElemType) where {T<:Real}
        # only 2d triangulations allowed yet
        new(coords,nodes,[ET],[],[[] []],[[] []],[],[],[[] []],[],[[] []],[[] []]);
    end
end



  # subtype for elem types that require 1D integration
  abstract type Abstract0DElemType <: AbstractElemType end
    struct ElemType0DPoint <: Abstract0DElemType end

  abstract type Abstract1DElemType <: AbstractElemType end
    include("GRIDelemtypes/1D_interval.jl");
  #  include("GRIDelemtypes/1D_2Dline.jl");

  # subtype for elem types that require 2D integration
  abstract type Abstract2DElemType <: AbstractElemType end
    include("GRIDelemtypes/2D_triangle.jl");
    include("GRIDelemtypes/2D_parallelogram.jl");

  # subtype for elem types that require 3D integration
  abstract type Abstract3DElemType <: AbstractElemType end


get_dimension(ET::Abstract0DElemType) = 0
get_dimension(ET::Abstract1DElemType) = 1
get_dimension(ET::Abstract2DElemType) = 2
get_dimension(ET::Abstract3DElemType) = 3


# show function for Grid
function show(Grid::Mesh)

    dim = size(Grid.nodes4cells,2) - 1;;
    nnodes = size(Grid.coords4nodes,1);
    ncells = size(Grid.nodes4cells,1);
    
	println("Mesh information");
	println("    dim : $(dim)")
	println(" nnodes : $(nnodes)")
	println(" ncells : $(ncells)")
end


function Mesh{T}(coords,nodes,ET::AbstractElemType,nrefinements) where {T<:Real}
    for j=1:nrefinements
        coords, nodes = uniform_refinement(ET,coords,nodes)
    end
    return Mesh{T}(coords,nodes,ET);
end


function Mesh{T}(coords,nodes,nodes4bfaces,bregions,ET::AbstractElemType,nrefinements) where {T<:Real}
    for j=1:nrefinements
        coords, nodes, nodes4bfaces, bregions = uniform_refinement(ET,coords,nodes)
    end
    grid = Mesh{T}(coords,nodes,ET)
    assign_boundaryregions!(grid,nodes4bfaces,bregions);
    return grid;
end

# default constructor for Float64-typed triangulations
function Mesh(coords,nodes,nrefinements = 0)
    for j=1:nrefinements
        coords, nodes, = uniform_refinement(coords,nodes)
    end
    return Mesh{Float64}(coords,nodes,ElemType2DTriangle());
end
  

function ensure_length4faces!(Grid::Mesh)
    ensure_nodes4faces!(Grid)
    celldim = size(Grid.nodes4faces,2) - 1;
    nfaces::Int = size(Grid.nodes4faces,1);
    if size(Grid.length4faces,1) != size(nfaces,1)
        if celldim == 1 # also allow d-dimensional points on a line!
            Grid.length4faces= zeros(eltype(Grid.coords4nodes),nfaces);
            xdim::Int = size(Grid.coords4nodes,2)
            for face = 1 : nfaces
                for d = 1 : xdim
                    Grid.length4faces[face] += (Grid.coords4nodes[Grid.nodes4faces[face,2],d] - Grid.coords4nodes[Grid.nodes4faces[face,1],d]).^2
                end
                 Grid.length4faces[face] = sqrt(Grid.length4faces[face]);    
            end   
        elseif celldim == 0 # points of length 1
            Grid.length4faces = ones(eltype(Grid.coords4nodes),nfaces);
        end
    end        
end  

function ensure_volume4cells!(Grid::Mesh)
    ncells::Int = size(Grid.nodes4cells,1);
    if size(Grid.volume4cells,1) != size(ncells,1)
        Grid.volume4cells = zeros(eltype(Grid.coords4nodes),ncells);
        if typeof(Grid.elemtypes[1]) <: ElemType1DInterval
            for cell = 1 : ncells
                Grid.volume4cells[cell] = abs(Grid.coords4nodes[Grid.nodes4cells[cell,2],1] - Grid.coords4nodes[Grid.nodes4cells[cell,1],1])
            end    
        elseif typeof(Grid.elemtypes[1]) <: Abstract1DElemType
            xdim::Int = size(Grid.coords4nodes,2)
            for cell = 1 : ncells
                for d = 1 : xdim
                    Grid.volume4cells[cell] += (Grid.coords4nodes[Grid.nodes4cells[cell,2],d] - Grid.coords4nodes[Grid.nodes4cells[cell,1],d]).^2
                end
                Grid.volume4cells[cell] = sqrt(Grid.volume4cells[cell]);    
            end    
        elseif typeof(Grid.elemtypes[1]) <: ElemType2DTriangle
            for cell = 1 : ncells 
                Grid.volume4cells[cell] = 1 // 2 * (
               Grid.coords4nodes[Grid.nodes4cells[cell,1],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,2],2] -  Grid.coords4nodes[Grid.nodes4cells[cell,3],2])
            + Grid.coords4nodes[Grid.nodes4cells[cell,2],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,3],2] - Grid.coords4nodes[Grid.nodes4cells[cell,1],2])
            + Grid.coords4nodes[Grid.nodes4cells[cell,3],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,1],2] - Grid.coords4nodes[Grid.nodes4cells[cell,2],2]));
            end        
        elseif typeof(Grid.elemtypes[1]) <: ElemType2DParallelogram
            for cell = 1 : ncells 
                Grid.volume4cells[cell] = (
               Grid.coords4nodes[Grid.nodes4cells[cell,1],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,2],2] -  Grid.coords4nodes[Grid.nodes4cells[cell,3],2])
            + Grid.coords4nodes[Grid.nodes4cells[cell,2],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,3],2] - Grid.coords4nodes[Grid.nodes4cells[cell,1],2])
            + Grid.coords4nodes[Grid.nodes4cells[cell,3],1] * (Grid.coords4nodes[Grid.nodes4cells[cell,1],2] - Grid.coords4nodes[Grid.nodes4cells[cell,2],2]));
            end        
       # elseif typeof(Grid.elemtypes[1]) <: ElemType3DTetraeder
       #    A = ones(eltype(Grid.coords4nodes),4,4);
       #    for cell = 1 : ncells 
       #     A[1,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,1],:];
       #     A[2,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,2],:];
       #     A[3,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,3],:];
       #     A[4,[2,3,4]] = Grid.coords4nodes[Grid.nodes4cells[cell,4],:];
       #     Grid.volume4cells[cell] = 1 // 6 * abs(det(A));
       #    end
        end
    end        
end   

# determine the face numbers of the boundary faces
# (they appear only once in faces4cells)
function ensure_bfaces!(Grid::Mesh)
    dim = get_dimension(Grid.elemtypes[1])
    @assert dim <= 3
    if size(Grid.bfaces,1) <= 0
        ensure_faces4cells!(Grid::Mesh)
        ncells = size(Grid.faces4cells,1);    
        nfaces = size(Grid.nodes4faces,1);
        takeface = zeros(Bool,nfaces);
        for cell = 1 : ncells
            for j = 1 : size(Grid.faces4cells,2)
                @inbounds takeface[Grid.faces4cells[cell,j]] = .!takeface[Grid.faces4cells[cell,j]];
            end    
        end
        Grid.bfaces = findall(takeface);
        Grid.bregions = ones(Int64,length(Grid.bfaces));
    end
end

function assign_boundaryregions!(grid,nodes4bfaces,bregions)
    ensure_bfaces!(grid)
    ensure_nodes4faces!(grid)
    nodes1 = [0, 0]
    nodes2 = [0, 0]
    for j = 1 : length(bregions)
        nodes1[:] = sort(nodes4bfaces[:,j])
        for k = 1 : length(grid.bfaces)
            nodes2[:] = sort(grid.nodes4faces[grid.bfaces[k],:])
            if (nodes1 == nodes2)
                grid.bregions[k] = bregions[j]
                break;
            end
        end        
    end
end

# compute nodes4faces (implicating an enumeration of the faces)
function ensure_nodes4faces!(Grid::Mesh)
    dim = get_dimension(Grid.elemtypes[1])
    @assert dim <= 3
    ncells::Int = size(Grid.nodes4cells,1);
    index::Int = 0;
    temp::Int64 = 0;
    if (size(Grid.nodes4faces,1) <= 0)
        if dim == 1
            # in 1D nodes are faces
            nnodes::Int = size(Grid.coords4nodes,1);
            Grid.nodes4faces = zeros(Int64,nnodes,1);
            Grid.nodes4faces[:] = 1:nnodes;
        elseif dim == 2
            nodes_per_cell = size(Grid.nodes4cells,2)
            helperfield = zeros(Int64,nodes_per_cell,1)
            helperfield[:] = 0:nodes_per_cell-1
            helperfield[1] = nodes_per_cell
            # compute nodes4faces with duplicates
            Grid.nodes4faces = zeros(Int64,nodes_per_cell*ncells,2);
            index = 1
            for cell = 1 : ncells
                for k = 1 : nodes_per_cell
                    Grid.nodes4faces[index,1] = Grid.nodes4cells[cell,k];
                    Grid.nodes4faces[index,2] = Grid.nodes4cells[cell,helperfield[k]];
                    index += 1;
                end    
            end    
    
            # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
            for j = 1 : nodes_per_cell*ncells
                if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
                    temp = Grid.nodes4faces[j,1];
                    Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = temp;
                end
            end
        
            # find unique rows -> this fixes the enumeration of the faces!
            Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);
        elseif dim == 3 # might work for tets
            # compute nodes4faces with duplicates
            Grid.nodes4faces = zeros(Int64,4*ncells,3);
            for cell = 1 : ncells
                Grid.nodes4faces[index+1,1] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+1,2] = Grid.nodes4cells[cell,2];
                Grid.nodes4faces[index+1,3] = Grid.nodes4cells[cell,3];
                Grid.nodes4faces[index+2,1] = Grid.nodes4cells[cell,2]; 
                Grid.nodes4faces[index+2,2] = Grid.nodes4cells[cell,3]; 
                Grid.nodes4faces[index+2,3] = Grid.nodes4cells[cell,4];
                Grid.nodes4faces[index+3,1] = Grid.nodes4cells[cell,3];
                Grid.nodes4faces[index+3,2] = Grid.nodes4cells[cell,4];
                Grid.nodes4faces[index+3,3] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+4,1] = Grid.nodes4cells[cell,4];
                Grid.nodes4faces[index+4,2] = Grid.nodes4cells[cell,1];
                Grid.nodes4faces[index+4,3] = Grid.nodes4cells[cell,2];
                index += 4;
            end    
    
            # sort each row ( faster than: sort!(Grid.nodes4faces, dims = 2);)
            for j = 1 : 4*ncells
                if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
                    temp = Grid.nodes4faces[j,1];
                    Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = temp;
                end
                if Grid.nodes4faces[j,3] > Grid.nodes4faces[j,2]
                    temp = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = Grid.nodes4faces[j,3];
                    Grid.nodes4faces[j,3] = temp;
                end
                if Grid.nodes4faces[j,2] > Grid.nodes4faces[j,1]
                    temp = Grid.nodes4faces[j,1];
                    Grid.nodes4faces[j,1] = Grid.nodes4faces[j,2];
                    Grid.nodes4faces[j,2] = temp;
                end
            end
        
            # find unique rows -> this fixes the enumeration of the faces!
            Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);    
        end
    end    
end

# compute faces4cells
function ensure_faces4cells!(Grid::Mesh)
    dim = get_dimension(Grid.elemtypes[1])
    @assert dim <= 3
    if size(Grid.faces4cells,1) != size(Grid.nodes4cells,1)
        ensure_nodes4faces!(Grid)
        if dim == 1
            # in 1D nodes are faces
            Grid.faces4cells = Grid.nodes4cells;        
        elseif dim == 2
            nodes_per_cell = size(Grid.nodes4cells,2)
            nnodes = size(Grid.coords4nodes,1);
            nfaces = size(Grid.nodes4faces,1);
            ncells = size(Grid.nodes4cells,1);
    
            face4nodes = sparse(view(Grid.nodes4faces,:,1),view(Grid.nodes4faces,:,2),1:nfaces,nnodes,nnodes);
            face4nodes = face4nodes + face4nodes';
    
            Grid.faces4cells = zeros(Int,size(Grid.nodes4cells,1),nodes_per_cell);
            helperfield = zeros(Int64,nodes_per_cell,1)
            helperfield[:] = 2:nodes_per_cell+1
            helperfield[end] = 1
            for cell = 1 : ncells
                for k = 1 : nodes_per_cell
                    Grid.faces4cells[cell,k] = face4nodes[Grid.nodes4cells[cell,k],Grid.nodes4cells[cell,helperfield[k]]];
                end    
            end
        elseif dim == 3    
            # todo
            println("faces4cells for tets not yet implemented!")
            @assert dim <= 3
        end
    end    
end


# compute signs4cells
function ensure_signs4cells!(Grid::Mesh)
    dim = get_dimension(Grid.elemtypes[1])
    @assert dim <= 3
    if size(Grid.signs4cells,1) != size(Grid.nodes4cells,1)
        ensure_faces4cells!(Grid)
        ncells::Int64 = size(Grid.nodes4cells,1)
        Grid.signs4cells = ones(ncells,size(Grid.faces4cells,2))
        for cell = 1 : ncells
            for f = 1 : size(Grid.faces4cells,2)
                if Grid.nodes4faces[Grid.faces4cells[cell,f],1] != Grid.nodes4cells[cell,f]
                    Grid.signs4cells[cell,f] = -1
                end       
            end
        end
    end   
end


# compute cells4faces
function ensure_cells4faces!(Grid::Mesh)
    dim = get_dimension(Grid.elemtypes[1])
    @assert dim <= 3
    ensure_nodes4faces!(Grid)
    nfaces::Int = size(Grid.nodes4faces,1);
    ensure_faces4cells!(Grid)
    if size(Grid.cells4faces,1) != nfaces
        Grid.cells4faces = zeros(Int,nfaces,2);
        for j = 1:size(Grid.faces4cells,1) 
            for k = 1:size(Grid.faces4cells,2)
                if Grid.cells4faces[Grid.faces4cells[j,k],1] == 0
                    Grid.cells4faces[Grid.faces4cells[j,k],1] = j
                else    
                    Grid.cells4faces[Grid.faces4cells[j,k],2] = j
                end    
            end
        end
    end
end


# compute normal4faces
function ensure_normal4faces!(Grid::Mesh)
    dim::Int = size(Grid.nodes4cells,2)
    ensure_nodes4faces!(Grid)
    xdim::Int = size(Grid.coords4nodes,2)
    nfaces::Int = size(Grid.nodes4faces,1);
    if size(Grid.normal4faces,1) != nfaces
        Grid.normal4faces = zeros(Int,nfaces,xdim);
        for j = 1:nfaces 
            # rotate tangent
            Grid.normal4faces[j,:] = Grid.coords4nodes[Grid.nodes4faces[j,1],[2,1]] - Grid.coords4nodes[Grid.nodes4faces[j,2],[2,1]];
            Grid.normal4faces[j,1] *= -1
            # divide by length
            Grid.normal4faces[j,:] ./= sqrt(dot(Grid.normal4faces[j,:],Grid.normal4faces[j,:]))
            
        end
    end
end


function get_boundary_grid(Grid::Mesh);
    ensure_nodes4faces!(Grid)
    ensure_bfaces!(Grid)
    return Mesh(Grid.coords4nodes,Grid.nodes4faces[Grid.bfaces,:]);
end

end # module
