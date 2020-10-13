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
abstract type ON_IFACES <: ON_FACES end  # facedofs on interior faces
abstract type ON_BFACES <: AbstractAssemblyType end # facedofs on boundary faces
abstract type ON_EDGES <: AbstractAssemblyType end  # edgedofs on all edges

GridComponentNodes4AssemblyType(::Type{ON_CELLS}) = CellNodes
GridComponentNodes4AssemblyType(::Type{<:ON_FACES}) = FaceNodes
GridComponentNodes4AssemblyType(::Type{ON_BFACES}) = BFaceNodes
GridComponentNodes4AssemblyType(::Type{<:ON_EDGES}) = EdgeNodes

GridComponentVolumes4AssemblyType(::Type{ON_CELLS}) = CellVolumes
GridComponentVolumes4AssemblyType(::Type{<:ON_FACES}) = FaceVolumes
GridComponentVolumes4AssemblyType(::Type{ON_BFACES}) = BFaceVolumes
GridComponentVolumes4AssemblyType(::Type{<:ON_EDGES}) = EdgeVolumes

GridComponentGeometries4AssemblyType(::Type{ON_CELLS}) = CellGeometries
GridComponentGeometries4AssemblyType(::Type{<:ON_FACES}) = FaceGeometries
GridComponentGeometries4AssemblyType(::Type{ON_BFACES}) = BFaceGeometries
GridComponentGeometries4AssemblyType(::Type{<:ON_EDGES}) = EdgeGeometries

GridComponentUniqueGeometries4AssemblyType(::Type{ON_CELLS}) = UniqueCellGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{<:ON_FACES}) = UniqueFaceGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{ON_BFACES}) = UniqueBFaceGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{<:ON_EDGES}) = UniqueEdgeGeometries

GridComponentRegions4AssemblyType(::Type{ON_CELLS}) = CellRegions
GridComponentRegions4AssemblyType(::Type{<:ON_FACES}) = FaceRegions
GridComponentRegions4AssemblyType(::Type{ON_BFACES}) = BFaceRegions
GridComponentRegions4AssemblyType(::Type{<:ON_EDGES}) = EdgeRegions