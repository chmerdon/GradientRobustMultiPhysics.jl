 
#############################
# Finite Element Supertypes #
#############################
#
# they are used to steer the kind of local2global transformation
# below subtypes are defined that define basis functions on reference geometries 
# and some other information like polyonomial degrees etc.

abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
abstract type AbstractH1FiniteElementWithCoefficients <: AbstractH1FiniteElement end
abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end


#########################
# Finite Element Spaces #
#########################
#
# struct that has a finite element subtype (see below) as parameter
# and carries dofmaps, grid information and access to coefficients
# (links have to be stored here to avoid search in xgrid dicts)
#
mutable struct FESpace{FEType<:AbstractFiniteElement}
    name::String                          # full name of finite element space (used in messages)
    ndofs::Int                            # total number of dofs
    xgrid::ExtendableGrid                 # link to xgrid 
    CellDofs::VariableTargetAdjacency     # place to save cell dofs (filled by constructor)
    FaceDofs::VariableTargetAdjacency     # place to save face dofs (filled by constructor)
    BFaceDofs::VariableTargetAdjacency    # place to save bface dofs (filled by constructor)
    xFaceNormals::Array{Float64,2}        # link to coefficient values
    xFaceVolumes::Array{Float64,1}        # link to coefficient values
    xCellFaces::VariableTargetAdjacency   # link to coefficient indices
    xCellSigns::VariableTargetAdjacency   # place to save cell signumscell coefficients
end

# constructor
function FESpace{FEType}(xgrid::ExtendableGrid; dofmap_needed = true) where {FEType <:AbstractFiniteElement}
    # first generate some empty FESpace
    dummyVTA = VariableTargetAdjacency(Int32)
    FES = FESpace{FEType}("",0,xgrid,dummyVTA,dummyVTA,dummyVTA,Array{Float64,2}(undef,0,0),Array{Float64,1}(undef,0),dummyVTA,dummyVTA)

    # then update data according to init specifications in FEdefinition files
    init!(FES; dofmap_needed = dofmap_needed)

    return FES
end

Base.eltype(::Type{FESpace{FEType}}) where {FEType} = FEType

# show function for FiniteElementSpace
function Base.show(io::IO, FES::FESpace{FEType}) where {FEType}
	  println("\nFESpace information")
    println("===================")
    println("     name = $(FES.name)")
    println("   FEType = $FEType")
    println("  FEClass = $(supertype(FEType))")
    println("    ndofs = $(FES.ndofs)\n")
    if num_sources(FES.CellDofs) > 0
        println("dofmap for CELL available (nitems=$(num_sources(FES.CellDofs)), maxdofs4item=$(max_num_targets_per_source(FES.CellDofs)))")
    else
        println("dofmap for CELL not available")
    end
    if num_sources(FES.FaceDofs) > 0
        println("dofmap for FACE available (nitems=$(num_sources(FES.FaceDofs)), maxdofs4item=$(max_num_targets_per_source(FES.FaceDofs)))")
    else
        println("dofmap for FACE not available")
    end
    if num_sources(FES.BFaceDofs) > 0
        println("dofmap for BFACE available (nitems=$(num_sources(FES.BFaceDofs)), maxdofs4item=$(max_num_targets_per_source(FES.BFaceDofs)))")
    else
        println("dofmap for BFACE not available")
    end
end

###########################
# Finite Element Subtypes #
###########################
#
# subtypes of the finite element supertypes above
# each defined in its own file
  
# Hdiv-conforming elements (only vector-valued)
# lowest order
include("FEdefinitions/Hdiv_RT0.jl");

# H1 conforming elements (also Crouzeix-Raviart)
# lowest order
include("FEdefinitions/H1_P1.jl");
include("FEdefinitions/H1_MINI.jl");
include("FEdefinitions/H1nc_CR.jl");
include("FEdefinitions/H1v_BR.jl"); # Bernardi--Raugel (only vector-valued, with coefficients)
# second order
include("FEdefinitions/H1_P2.jl");

# L2 conforming elements
include("FEdefinitions/L2_P0.jl"); # currently masked as H1 element

# Hcurl-conforming elements
# TODO

