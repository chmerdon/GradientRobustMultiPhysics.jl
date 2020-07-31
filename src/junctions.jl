########################
# AbstractAssemblyType #
########################

# this type is used to steer where certain things live and assemble on
# mainly if it lives on CELLs, FACEs or BFACEs
#
# todo : in 3D we will also need EDGEs
# 
# mixed types like BFACECELL allow to evaluate cell dofs on bfaces (wip)

abstract type AbstractAssemblyType end
abstract type AssemblyTypeCELL <: AbstractAssemblyType end  # celldofs on all cells 
abstract type AssemblyTypeFACE <: AbstractAssemblyType end  # facedofs on all faces
abstract type AssemblyTypeBFACE <: AbstractAssemblyType end # facedofs on bfaces
abstract type AssemblyTypeBFACECELL <: AbstractAssemblyType end # celldofs on bfaces
#abstract type AssemblyTypeEDGE end

GridComponentNodes4AssemblyType(::Type{AssemblyTypeCELL}) = CellNodes
GridComponentNodes4AssemblyType(::Type{AssemblyTypeFACE}) = FaceNodes
GridComponentNodes4AssemblyType(::Type{AssemblyTypeBFACE}) = BFaceNodes
GridComponentNodes4AssemblyType(::Type{AssemblyTypeBFACECELL}) = CellNodes

GridComponentVolumes4AssemblyType(::Type{AssemblyTypeCELL}) = CellVolumes
GridComponentVolumes4AssemblyType(::Type{AssemblyTypeFACE}) = FaceVolumes
GridComponentVolumes4AssemblyType(::Type{AssemblyTypeBFACE}) = BFaceVolumes
GridComponentVolumes4AssemblyType(::Type{AssemblyTypeBFACECELL}) = BFaceVolumes

GridComponentGeometries4AssemblyType(::Type{AssemblyTypeCELL}) = CellGeometries
GridComponentGeometries4AssemblyType(::Type{AssemblyTypeFACE}) = FaceGeometries
GridComponentGeometries4AssemblyType(::Type{AssemblyTypeBFACE}) = BFaceGeometries
GridComponentGeometries4AssemblyType(::Type{AssemblyTypeBFACECELL}) = BFaceGeometries

GridComponentRegions4AssemblyType(::Type{AssemblyTypeCELL}) = CellRegions
GridComponentRegions4AssemblyType(::Type{AssemblyTypeFACE}) = FaceRegions
GridComponentRegions4AssemblyType(::Type{AssemblyTypeBFACE}) = BFaceRegions
GridComponentRegions4AssemblyType(::Type{AssemblyTypeBFACECELL}) = BFaceRegions


# in situations where we integrate on faces but want to evaluate cell dofs we need
# to transform the xref on ech cellface (xrefFACE) to xref on the CELL;
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