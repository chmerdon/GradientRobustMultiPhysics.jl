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

ItemType4AssemblyType(::Type{ON_CELLS}) = ITEMTYPE_CELL
ItemType4AssemblyType(::Type{<:ON_FACES}) = ITEMTYPE_FACE
ItemType4AssemblyType(::Type{ON_BFACES}) = ITEMTYPE_BFACE
ItemType4AssemblyType(::Type{<:ON_EDGES}) = ITEMTYPE_EDGE
ItemType4AssemblyType(::Type{ON_BEDGES}) = ITEMTYPE_BEDGE

GridComponentNodes4AssemblyType(AT::Type{<:AbstractAssemblyType}) = GridComponent4TypeProperty(ItemType4AssemblyType(AT),PROPERTY_NODES)
GridComponentVolumes4AssemblyType(AT::Type{<:AbstractAssemblyType}) = GridComponent4TypeProperty(ItemType4AssemblyType(AT),PROPERTY_VOLUME)
GridComponentGeometries4AssemblyType(AT::Type{<:AbstractAssemblyType}) = GridComponent4TypeProperty(ItemType4AssemblyType(AT),PROPERTY_GEOMETRY)
GridComponentUniqueGeometries4AssemblyType(AT::Type{<:AbstractAssemblyType}) = GridComponent4TypeProperty(ItemType4AssemblyType(AT),PROPERTY_UNIQUEGEOMETRY)
GridComponentRegions4AssemblyType(AT::Type{<:AbstractAssemblyType}) = GridComponent4TypeProperty(ItemType4AssemblyType(AT),PROPERTY_REGION)

