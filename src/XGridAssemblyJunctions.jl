abstract type AbstractAssemblyType end
abstract type AbstractAssemblyTypeCELL <: AbstractAssemblyType end
abstract type AbstractAssemblyTypeFACE <: AbstractAssemblyType end
abstract type AbstractAssemblyTypeBFACE <: AbstractAssemblyTypeFACE end # only boundary faces
#abstract type AbstractAssemblyTypeEDGE end

GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellNodes
GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceNodes
GridComponentNodes4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceNodes

GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellVolumes
GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceVolumes
GridComponentVolumes4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceVolumes

GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeCELL}) = CellTypes
GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeFACE}) = FaceTypes
GridComponentTypes4AssemblyType(::Type{AbstractAssemblyTypeBFACE}) = BFaceTypes

