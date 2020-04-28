abstract type AbstractAssemblyType end
abstract type AbstractAssemblyTypeCELL <: AbstractAssemblyType end  # celldofs on all cells 
abstract type AbstractAssemblyTypeFACE <: AbstractAssemblyType end  # facedofs on all faces
abstract type AbstractAssemblyTypeBFACE <: AbstractAssemblyType end # facedofs on bfaces
abstract type AbstractAssemblyTypeBFACECELL <: AbstractAssemblyType end # celldofs on bfaces
#abstract type AbstractAssemblyTypeEDGE end

GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellNodes
GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceNodes
GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceNodes
GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeBFACECELL}) = CellNodes

GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellVolumes
GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceVolumes
GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceVolumes
GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeBFACECELL}) = BFaceVolumes

GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellGeometries
GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceGeometries
GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceGeometries
GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeBFACECELL}) = BFaceGeometries


# in situations where we integrate of faces but want to evaluate cell dofs we need
# to transform the xref on the face (xrefFACE) to xref on the CELL;
# this transformation depends on the geometry and is specified below

xrefFACE2xrefCELL(::Type{<:Edge1D}) = [ (xref4FACE) -> [1],
                                        (xref4FACE) -> [1] ]

xrefFACE2xrefCELL(::Type{<:Triangle2D}) = [ (xref4FACE) -> [xref4FACE[1],0],
                                            (xref4FACE) -> [1-xref4FACE[1],xref4FACE[1]], 
                                            (xref4FACE) -> [0,1-xref4FACE[1]] ]

xrefFACE2xrefCELL(::Type{<:Parallelogram2D}) = [ (xref4FACE) -> [xref4FACE[1],0],
                                                 (xref4FACE) -> [1,xref4FACE[1]], 
                                                 (xref4FACE) -> [1-xref4FACE[1],1], 
                                                 (xref4FACE) -> [0,1-xref4FACE[1]] ]