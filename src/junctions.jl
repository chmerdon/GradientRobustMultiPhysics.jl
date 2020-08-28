########################
# AbstractAssemblyType #
########################

# this type is used to steer where certain things live and assemble on
# mainly if it lives on CELLs, FACEs or BFACEs
#
# todo : in 3D we will also need EDGEs

abstract type AbstractAssemblyType end
abstract type ON_CELLS <: AbstractAssemblyType end  # celldofs on all cells 
abstract type ON_FACES <: AbstractAssemblyType end  # facedofs on all faces
abstract type ON_BFACES <: AbstractAssemblyType end # facedofs on bfaces

GridComponentNodes4AssemblyType(::Type{ON_CELLS}) = CellNodes
GridComponentNodes4AssemblyType(::Type{ON_FACES}) = FaceNodes
GridComponentNodes4AssemblyType(::Type{ON_BFACES}) = BFaceNodes

GridComponentVolumes4AssemblyType(::Type{ON_CELLS}) = CellVolumes
GridComponentVolumes4AssemblyType(::Type{ON_FACES}) = FaceVolumes
GridComponentVolumes4AssemblyType(::Type{ON_BFACES}) = BFaceVolumes

GridComponentGeometries4AssemblyType(::Type{ON_CELLS}) = CellGeometries
GridComponentGeometries4AssemblyType(::Type{ON_FACES}) = FaceGeometries
GridComponentGeometries4AssemblyType(::Type{ON_BFACES}) = BFaceGeometries

GridComponentUniqueGeometries4AssemblyType(::Type{ON_CELLS}) = UniqueCellGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{ON_FACES}) = UniqueFaceGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{ON_BFACES}) = UniqueBFaceGeometries

GridComponentRegions4AssemblyType(::Type{ON_CELLS}) = CellRegions
GridComponentRegions4AssemblyType(::Type{ON_FACES}) = FaceRegions
GridComponentRegions4AssemblyType(::Type{ON_BFACES}) = BFaceRegions