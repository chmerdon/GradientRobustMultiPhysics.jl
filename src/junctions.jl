########################
# AbstractAssemblyType #
########################

# this type is used to steer where certain things live and assemble on
# mainly if it lives on CELLs, FACEs or BFACEs

abstract type AbstractAssemblyType end 
abstract type AT_NODES <: AbstractAssemblyType end  # at nodes (only available for H1 conforming interpolation)
abstract type ON_CELLS <: AbstractAssemblyType end  # on all cells 
abstract type ON_FACES <: AbstractAssemblyType end  # on all faces
abstract type ON_IFACES <: ON_FACES end  # on interior faces
abstract type ON_BFACES <: AbstractAssemblyType end # on boundary faces
abstract type ON_EDGES <: AbstractAssemblyType end  # on all edges
abstract type ON_BEDGES <: AbstractAssemblyType end # on boundary edges

GridComponentNodes4AssemblyType(::Type{ON_CELLS}) = CellNodes
GridComponentNodes4AssemblyType(::Type{<:ON_FACES}) = FaceNodes
GridComponentNodes4AssemblyType(::Type{ON_BFACES}) = BFaceNodes
GridComponentNodes4AssemblyType(::Type{<:ON_EDGES}) = EdgeNodes
GridComponentNodes4AssemblyType(::Type{ON_BEDGES}) = BEdgeNodes

GridComponentVolumes4AssemblyType(::Type{ON_CELLS}) = CellVolumes
GridComponentVolumes4AssemblyType(::Type{<:ON_FACES}) = FaceVolumes
GridComponentVolumes4AssemblyType(::Type{ON_BFACES}) = BFaceVolumes
GridComponentVolumes4AssemblyType(::Type{<:ON_EDGES}) = EdgeVolumes
GridComponentVolumes4AssemblyType(::Type{ON_BEDGES}) = BEdgeVolumes

GridComponentGeometries4AssemblyType(::Type{ON_CELLS}) = CellGeometries
GridComponentGeometries4AssemblyType(::Type{<:ON_FACES}) = FaceGeometries
GridComponentGeometries4AssemblyType(::Type{ON_BFACES}) = BFaceGeometries
GridComponentGeometries4AssemblyType(::Type{<:ON_EDGES}) = EdgeGeometries
GridComponentGeometries4AssemblyType(::Type{ON_BEDGES}) = BEdgeGeometries

GridComponentUniqueGeometries4AssemblyType(::Type{ON_CELLS}) = UniqueCellGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{<:ON_FACES}) = UniqueFaceGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{ON_BFACES}) = UniqueBFaceGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{<:ON_EDGES}) = UniqueEdgeGeometries
GridComponentUniqueGeometries4AssemblyType(::Type{ON_BEDGES}) = UniqueBEdgeGeometries

GridComponentRegions4AssemblyType(::Type{ON_CELLS}) = CellRegions
GridComponentRegions4AssemblyType(::Type{<:ON_FACES}) = FaceRegions
GridComponentRegions4AssemblyType(::Type{ON_BFACES}) = BFaceRegions
GridComponentRegions4AssemblyType(::Type{<:ON_EDGES}) = EdgeRegions
GridComponentRegions4AssemblyType(::Type{ON_BEDGES}) = BEdgeRegions