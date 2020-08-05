#######################################################################################################
#######################################################################################################
### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
#######################################################################################################
#######################################################################################################

abstract type AbstractFiniteElement end
  

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


"""
$(TYPEDEF)

A struct that has a finite element type as parameter and carries dofmaps (CellDofs, FaceDofs, BFaceDofs) plus additional grid information and access to arrays holding coefficients if needed.
"""
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
    xCellFaceSigns::VariableTargetAdjacency   # place to save cell signumscell coefficients
end

function FESpace{FEType}(xgrid::ExtendableGrid; name = "", dofmaps_needed = [AssemblyTypeCELL, AssemblyTypeFACE, AssemblyTypeBFACE], verbosity = 0 ) where {FEType <:AbstractFiniteElement}
    # first generate some empty FESpace
    dummyVTA = VariableTargetAdjacency(Int32)
    FES = FESpace{FEType}(name,0,xgrid,dummyVTA,dummyVTA,dummyVTA,Array{Float64,2}(undef,0,0),Array{Float64,1}(undef,0),dummyVTA,dummyVTA)

    if verbosity > 0
        println("  Initialising FESpace $FEType...")
    end
    # then update data according to init specifications in FEdefinition files
    init!(FES)

    # generate required dof maps
    for j = 1 : length(dofmaps_needed)
        if verbosity > 0
            println("  ...generating dofmap for $(dofmaps_needed[j])")
            @time init_dofmap!(FES, dofmaps_needed[j])
        else
            init_dofmap!(FES, dofmaps_needed[j])
        end
    end

    return FES
end

function FESpace(name::String, length::Int)
    # first generate some empty FESpace
    dummyVTA = VariableTargetAdjacency(Int32)
    FES = FESpace{AbstractFiniteElement}(name,length,ExtendableGrid{Float64,Int32}(),dummyVTA,dummyVTA,dummyVTA,Array{Float64,2}(undef,0,0),Array{Float64,1}(undef,0),dummyVTA,dummyVTA)
    return FES
end

"""
$(TYPEDSIGNATURES)

Custom `eltype` function for `FESpace` returns the finite element type of the finite element space.
"""
Base.eltype(::FESpace{FEType}) where {FEType<:AbstractFiniteElement} = FEType


"""
$(TYPEDSIGNATURES)

Custom `show` function for `FESpace` that prints some information and all available dofmaps.
"""
function Base.show(io::IO, FES::FESpace{FEType}) where {FEType<:AbstractFiniteElement}
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
include("fedefs/hdiv_rt0.jl");
include("fedefs/hdiv_bdm1.jl");

# H1 conforming elements (also Crouzeix-Raviart)
# lowest order
include("fedefs/h1_p1.jl");
include("fedefs/h1_mini.jl");
include("fedefs/h1nc_cr.jl");
include("fedefs/h1v_br.jl"); # Bernardi--Raugel (only vector-valued, with coefficients)
# second order
include("fedefs/h1_p2.jl");

# L2 conforming elements
include("fedefs/l2_p0.jl"); # currently masked as H1 element
include("fedefs/l2_p1.jl"); # currently masked as H1 element

# Hcurl-conforming elements
# TODO




# dummy function
get_polynomialorder(::Type{<:AbstractFiniteElement}, ::Type{<:Vertex0D}) = 0;
