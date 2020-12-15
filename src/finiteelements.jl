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
abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end


"""
````
mutable struct FESpace{FEType<:AbstractFiniteElement}
    name::String                          # full name of finite element space (used in messages)
    broken::Bool                          # if true, broken dofmaps are generated
    ndofs::Int                            # total number of dofs
    xgrid::ExtendableGrid                 # link to xgrid 
    dofmaps::Dict{Type{<:AbstractGridComponent},Any} # backpack with dofmaps
end
````

A struct that has a finite element type as parameter and carries dofmaps (CellDofs, FaceDofs, BFaceDofs) plus additional grid information and access to arrays holding coefficients if needed.
"""
mutable struct FESpace{FEType<:AbstractFiniteElement}
    name::String                          # full name of finite element space (used in messages)
    broken::Bool                          # if true, broken dofmaps are generated
    ndofs::Int                            # total number of dofs
    xgrid::ExtendableGrid                 # link to xgrid 
    dofmaps::Dict{Type{<:AbstractGridComponent},Any} # backpack with dofmaps
end

include("dofmaps.jl")

"""
$(TYPEDSIGNATURES)
Set new dofmap
"""
Base.setindex!(FES::FESpace,v,DM::Type{<:DofMap}) = FES.dofmaps[DM] = v



"""
````
function FESpace{FEType<:AbstractFiniteElement}(
    xgrid::ExtendableGrid;
    name = "",
    broken::Bool = false,
    verbosity = 0)
````

Constructur for FESpace. The broken switch allows to generate a broken finite element space (with no continuities at all). If no name is provided
it is generated automatically from FEType.
"""
function FESpace{FEType}(
    xgrid::ExtendableGrid;
    name = "",
    dofmaps_needed = "auto",
    broken::Bool = false,
    verbosity = 0 ) where {FEType <:AbstractFiniteElement}
    # piecewise constants are always broken
    if FEType <: H1P0 
        broken = true
    end
    
    # first generate some empty FESpace
    dummyVTA = VariableTargetAdjacency(Int32)
    FES = FESpace{FEType}(name,broken,0,xgrid,Dict{Type{<:AbstractGridComponent},Any}())

    if verbosity > 0
        println("  Initialising FESpace $FEType...")
    end
    if name == ""
        FES.name = broken ? "$FEType (broken)" : "$FEType"
    else
        FES.name = name
    end

    # count ndofs
    count_ndofs!(FES)

    # generate ordered dofmaps
    if dofmaps_needed != "auto"
        # generate required dof maps
        for j = 1 : length(dofmaps_needed)
            if verbosity > 0
                println("  ...generating dofmap for $(dofmaps_needed[j])")
                @time init_dofmap!(FES, dofmaps_needed[j])
            else
                init_dofmap!(FES, dofmaps_needed[j])
            end
        end
    else
        # dofmaps are generated on demand
    end

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
    println("")
    println("DofMaps");
    println("==========");
    for tuple in FES.dofmaps
        println("> $(tuple[1])")
    end
end

# default coefficient functions that can be overwritten by finite element that has non-default coeffcients
# ( see e.g. h1v_br.jl )
function get_coefficients_on_face!(FE::FESpace{<:AbstractFiniteElement}, ::Type{<:AbstractElementGeometry})
    function closure(coefficients, face)
        fill!(coefficients,1.0)
    end
end    
function get_coefficients_on_cell!(FE::FESpace{<:AbstractFiniteElement}, ::Type{<:AbstractElementGeometry})
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
    end
end    
function get_basissubset_on_cell!(FE::FESpace{<:AbstractFiniteElement}, ::Type{<:AbstractElementGeometry})
    function closure(subset_ids, cell)
        # it is assumed that subset_ids is already 1 vector of form subset_ids = 1:ndofs
        # meaning that all basis functions on the reference cells are used
        # see 3D implementation of BDM1 for an example how this is can be used the choose
        # different basis functions depending on the face orientations (which in 3D is not just a sign)
        return nothing
    end
end  
get_ndofs_on_cell_all(FEType::Type{<:AbstractFiniteElement}, EG::Type{<:AbstractElementGeometry}) = get_ndofs_on_cell(FEType, EG)


"""
$(TYPEDSIGNATURES)
To be called by getindex. This triggers lazy creation of 
non-existing dofmaps
"""
Base.get!(FES::FESpace,DM::Type{<:DofMap}) = get!( ()-> init_dofmap!(FES,DM), FES.dofmaps ,DM)

"""
````
Base.getindex(FES::FESpace,DM::Type{<:DofMap})
````
Generic method for obtaining dofmap.
This method is mutating in the sense that non-existing dofmaps
are created on demand.
Due to the fact that components are stored as Any the return
value triggers type instability.
"""
Base.getindex(FES::FESpace,DM::Type{<:DofMap})=get!(FES,DM)


function count_ndofs!(FES::FESpace{FEType}) where {FEType <: AbstractFiniteElement}
    xgrid = FES.xgrid
    EG = xgrid[UniqueCellGeometries]
    xCellGeometries = xgrid[CellGeometries]
    ncells4EG::Array{Tuple{DataType,Int64},1} = [(i, count(==(i), xCellGeometries)) for i in EG]
    ndofs4EG = zeros(Int, length(EG))
    ncomponents::Int = get_ncomponents(FEType)
    dofs4item4component = zeros(Int,4)
    dofs4item_single = zeros(Int,4)
    maxdofs4item4component = zeros(Int,4)
    maxdofs4item_single = zeros(Int,4)
    quantifier::Int = 0
    pattern_char::Char = ' '
    highest_quantifierNCEF = zeros(Int,0)
    totaldofs = 0
    for j = 1 : length(EG)
        pattern = get_dofmap_pattern(FEType, CellDofs, EG[j])
        for k = 1 : Int(length(pattern)/2)
            pattern_char = pattern[2*k-1]
            quantifier = parse(Int,pattern[2*k])
            if pattern_char == 'N'
                dofs4item4component[1] += quantifier
            elseif pattern_char == 'F'
                dofs4item4component[2] += quantifier
            elseif pattern_char == 'E'
                dofs4item4component[3] += quantifier
            elseif pattern_char == 'I'
                dofs4item4component[4] += quantifier
            elseif pattern_char == 'n'
                dofs4item_single[1] += quantifier
            elseif pattern_char == 'f'
                dofs4item_single[2] += quantifier
            elseif pattern_char == 'e'
                dofs4item_single[3] += quantifier
            elseif pattern_char == 'i'
                dofs4item_single[4] += quantifier
            end
        end
        ndofs4EG[j] = nnodes_for_geometry(EG[j]) * (ncomponents * dofs4item4component[1] + dofs4item_single[1])
        ndofs4EG[j] += nfaces_for_geometry(EG[j]) * (ncomponents * dofs4item4component[2] + dofs4item_single[2])
        ndofs4EG[j] += nedges_for_geometry(EG[j]) * (ncomponents * dofs4item4component[3] + dofs4item_single[3])
        ndofs4EG[j] += ncomponents * dofs4item4component[4] + dofs4item_single[4]
        if FES.broken == true
            totaldofs += ndofs4EG[j] * ncells4EG[j][2]
        else
            # only count interior dofs on cell here
            totaldofs +=  ncells4EG[j][2] * (ncomponents * dofs4item4component[4] + dofs4item_single[4])
            for n = 1 : 4
                maxdofs4item4component[n] = max(maxdofs4item4component[n], dofs4item4component[n])
                maxdofs4item_single[n] = max(maxdofs4item_single[n], dofs4item_single[n])
            end
        end
        fill!(dofs4item4component,0)
        fill!(dofs4item_single,0)
    end
    if FES.broken == false
        # add continuous dofs here
        if maxdofs4item4component[1] + maxdofs4item_single[1] > 0
            totaldofs += size(xgrid[Coordinates],2) * (ncomponents * maxdofs4item4component[1] + maxdofs4item_single[1])
        end
        if maxdofs4item4component[2] + maxdofs4item_single[2] > 0
            totaldofs += num_sources(xgrid[FaceNodes]) * (ncomponents * maxdofs4item4component[2] + maxdofs4item_single[2])
        end
        if maxdofs4item4component[3] + maxdofs4item_single[3] > 0
            totaldofs += num_sources(xgrid[EdgeNodes]) * (ncomponents * maxdofs4item4component[3] + maxdofs4item_single[3])
        end
    end
    FES.ndofs = totaldofs
end


include("fevector.jl");
include("fematrix.jl");
include("interpolations.jl")


# dummy functions
get_edim(FEType::Type{<:AbstractFiniteElement}) = 0 # not defined
get_polynomialorder(::Type{<:AbstractFiniteElement}, ::Type{<:Vertex0D}) = 0;


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
include("fedefs/hdiv_rt1.jl");

# H1 conforming elements (also Crouzeix-Raviart)
# lowest order
include("fedefs/h1_p0.jl");
include("fedefs/h1_p1.jl");
include("fedefs/h1_mini.jl");
include("fedefs/h1nc_cr.jl");
include("fedefs/h1v_br.jl"); # Bernardi--Raugel (only vector-valued, with coefficients)
# second order
include("fedefs/h1_p2.jl");
include("fedefs/h1_p2b.jl");

# Hcurl-conforming elements
include("fedefs/hcurl_n0.jl");




