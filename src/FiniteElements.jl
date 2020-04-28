module FiniteElements

using Grid # remove if switch to ExtendableGrid is completed
using ExtendableGrids
using FEXGrid
using LinearAlgebra
using SparseArrays
using Quadrature
using ForwardDiff

export AstractFiniteElement, AbstractH1FiniteElementWithCoefficients, AbstractH1FiniteElement, AbstractHdivFiniteElement
export FEFunction

abstract type AbstractFiniteElement end

  # subtype for Hdiv-conforming elements
  abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/HDIV_RT0.jl");
    include("FEdefinitions/HDIV_ABF0.jl");

    # second order
    include("FEdefinitions/HDIV_RT1.jl");
    include("FEdefinitions/HDIV_BDM1.jl");

  # subtype for H1 conforming elements (also Crouzeix-Raviart)
  abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/H1_P1.jl");
    include("FEdefinitions/H1_Q1.jl");
    include("FEdefinitions/H1_MINI.jl");
    include("FEdefinitions/H1_CR.jl");
    # second order
    include("FEdefinitions/H1_P2.jl");
    include("FEdefinitions/H1_P2B.jl");

    abstract type AbstractH1FiniteElementWithCoefficients <: AbstractH1FiniteElement end
       include("FEdefinitions/H1_BR.jl");

 
  # subtype for L2 conforming elements
  abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/L2_P0.jl");
    include("FEdefinitions/L2_P1.jl");
 
  # subtype for Hcurl-conforming elements
  abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end
    # TODO

struct FEFunction{T} <: AbstractVector{T}
    name::String
    FEType::AbstractFiniteElement
    coefficients::Vector{T}
end

function FEFunction{T}(name::String, FEType::AbstractFiniteElement) where T <: Real
    coefficients = zeros(T,FEType.ndofs)
end

Base.getindex(FEF::FEFunction,i)=FEFunction.Coefficients[i]

#### OLD STUFF

export AbstractFiniteElement, AbstractH1FiniteElement, AbstractHdivFiniteElement, AbstractHdivRTFiniteElement, AbstractHdivBDMFiniteElement, AbstractHcurlFiniteElement

 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################

# generate sparse identity matrices where some diagonal entries are set to zero
function speyec(n::Int,zd::Vector)
    Ic = SparseMatrixCSC(I,n,n)
    for j=1 : length(zd)
        Ic[zd[j],zd[j]] = 0.0
    end
    return Ic
end


# top level abstract types
abstract type AbstractFEOperator end

    

# basis function updates on cells and faces (to multiply e.g. by normal vector or reconstruction coefficients etc.)
function get_basis_coefficients_on_cell!(coefficients, FE::AbstractFiniteElement, cell::Int64, ::Grid.AbstractElemType)
    # default is one, replace if needed for special FE
    fill!(coefficients,1.0)
end    
function get_basis_coefficients_on_face!(coefficients, FE::AbstractFiniteElement, face::Int64, ::Grid.AbstractElemType)
    # default is one, replace if needed for special FE
    fill!(coefficients,1.0)
end    

function Hdivreconstruction_available(FE::AbstractFiniteElement)
    return false
end


# show function for FiniteElement
function show(FE::AbstractFiniteElement)
    nd = get_ndofs(FE);
    ET = FE.grid.elemtypes[1];
	mdc = get_ndofs4elemtype(FE,ET);
	mdf = get_ndofs4elemtype(FE,Grid.get_face_elemtype(ET));
	po = get_polynomial_order(FE);

	println("FiniteElement information");
	println("         name : " * FE.name);
	println("        ndofs : $(nd)")
	println("    polyorder : $(po)");
	println("  maxdofs c/f : $(mdc)/$(mdf)")
end


# show function for FiniteElement
function show_new(FE::AbstractFiniteElement)
	println("FiniteElement information")
	println("         name : " * FE.name)
	println("        ndofs : $(FE.ndofs)")
end

function show_dofmap(FE::AbstractFiniteElement)
    println("FiniteElement cell dofmap for " * FE.name);
    ET = FE.grid.elemtypes[1];
    ETF = Grid.get_face_elemtype(ET);
    ndofs4cell = FiniteElements.get_ndofs4elemtype(FE, ET)
    dofs = zeros(Int64,ndofs4cell)
	for cell = 1 : size(FE.grid.nodes4cells,1)
        print(string(cell) * " | ");
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET)
        for j = 1 : ndofs4cell
            print(string(dofs[j]) * " ");
        end
        println("");
	end
	
	println("\nFiniteElement face dofmap for " * FE.name);
    ndofs4face = FiniteElements.get_ndofs4elemtype(FE, ETF)
    dofs2 = zeros(Int64,ndofs4face)
	for face = 1 : size(FE.grid.nodes4faces,1)
        print(string(face) * " | ");
        FiniteElements.get_dofs_on_face!(dofs2, FE, face, ETF)
        for j = 1 : ndofs4face
            print(string(dofs2[j]) * " ");
        end
        println("");
	end
end

function string2FE(fem::String, grid::Grid.Mesh, dim::Int, ncomponents::Int = 1)
    if fem == "P1"
        # piecewise linear (continuous)
        FE = FiniteElements.getP1FiniteElement(grid,ncomponents);
    elseif fem == "Q1"    
        # piecewise linear on quads
        FE = FiniteElements.getQ1FiniteElement(grid,ncomponents);
    elseif fem == "P0"    
        # piecewise constant
        FE = FiniteElements.getP0FiniteElement(grid,ncomponents);
    elseif fem == "P1dc"
        # piecewise linear (disccontinuous)
        FE = FiniteElements.getP1discFiniteElement(grid,ncomponents);
    elseif fem == "P2"
        # piecewise quadratic (continuous)
        FE = FiniteElements.getP2FiniteElement(grid,ncomponents);
    elseif fem == "MINI"
        # MINI element (P1 + cell bubbles)
        FE = FiniteElements.getMINIFiniteElement(grid,dim,ncomponents);
    elseif fem == "CR"
        # Crouzeix--Raviart
        FE = FiniteElements.getCRFiniteElement(grid,dim,ncomponents);
    elseif fem == "BR"
        # Bernardi--Raugel
        FE = FiniteElements.getBRFiniteElement(grid,ncomponents);
    elseif fem == "P2B"
        # P2-bubble element (P2 + cell_bubbles in 2D/3D + face bubbles in 3D)
        FE = FiniteElements.getP2BFiniteElement(grid,dim,ncomponents);
    elseif fem == "RT0"
        # Raviart-Thomas 0
        FE = FiniteElements.getRT0FiniteElement(grid);
    elseif fem == "ABF0"
        # Arnold-Boffi-Falk 0
        FE = FiniteElements.getABF0FiniteElement(grid);
    elseif fem == "RT1"
        # Raviart-Thomas 1
        FE = FiniteElements.getRT1FiniteElement(grid);
    elseif fem == "BDM1"
        # Brezzi-Douglas-Marini 1
        FE = FiniteElements.getBDM1FiniteElement(grid);
    end    
end

# composite Finite elements (every component is a different FE)
include("FEdefinitions/COMPOSITE.jl");

# creates a zero vector of the correct length for the FE
function createFEVector(FE::AbstractFiniteElement)
    return zeros(get_ndofs(FE));
end

function vector_hessian(f, x)
    n = length(x)
    return ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
end

abstract type AbstractAssembleType end
abstract type AssembleTypeCELL <: AbstractAssembleType end
abstract type AssembleTypeFACE <: AbstractAssembleType end

mutable struct FEbasis_caller{FEType <: AbstractFiniteElement, AT<:AbstractAssembleType}
    FE::AbstractFiniteElement # link to FiniteElement
    ncomponents::Int64
    ndofs4item::Int64 
    refbasisvals::Array{Array{Float64,2},1} # basis evaluation on reference cell
    refgradients::Array{Float64,3} # derivatives on reference cell
    refgradients_2ndorder::Array{Float64,3} # 2nd order derivatives on reference cel
    coefficients::Array{Float64,2} # heap for coefficients
    Ahandler::Function  # function that generates matrix trafo
    Piolahandler::Function  # function that generates Piola trafo matrix
    A::Matrix{Float64}  # matrix trafo for gradients
    Piola::Matrix{Float64} # matrix trafo for Hdiv or Hcurl elements
    det::Float64        # determinant of matrix
    xdim::Int64
    offsets::Array{Int64,1}
    offsets2::Array{Int64,1}
    current_item::Int64
    current_dofs::Array{Int64,1}
    with_standard_derivs::Bool
    with_higher_derivs::Bool
end

function FEbasis_caller(FE::AbstractH1FiniteElement, qf::QuadratureFormula, with_standard_derivs::Bool, with_higher_derivs::Bool = false)
    ET = FE.grid.elemtypes[1]
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    refbasisvals = Array{Array{Float64,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        if ncomponents == 1
            refbasisvals[i] = reshape(FiniteElements.get_basis_on_cell(FE, ET)(qf.xref[i]),:,1)
        else
            refbasisvals[i] = FiniteElements.get_basis_on_cell(FE, ET)(qf.xref[i])
        end    
    end    
    ndofs4cell = size(refbasisvals[1],1)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    Ahandler = Grid.local2global_tinv_jacobian(FE.grid,ET)
    Piolahandler = (x) -> 0; #dummy function
    xdim = size(FE.grid.coords4nodes,2)
    A = zeros(Float64,xdim,xdim)
    Piola = zeros(Float64,0,0)
    offsets = 0:xdim:(ncomponents*xdim);
    offsets2 = 0:ndofs4cell:ncomponents*ndofs4cell;
    current_dofs = zeros(Int64,ndofs4cell)
    if with_standard_derivs == false
        refgradients = zeros(Float64,0,0,0)
    else
        refgradients = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,length(qf.xref[1]))
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refgradients[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_cell(FE, ET),qf.xref[i]);
        end 
    end

    if with_higher_derivs == false
        refgradients_2ndorder = zeros(Float64,0,0,0)
    else
        refgradients_2ndorder = zeros(Float64,length(qf.w),ncomponents*ndofs4cell*length(qf.xref[1]),length(qf.xref[1]))
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            #refgradients_2ndorder[i,:,:]
            refgradients_2ndorder[i,:,:] = vector_hessian(FiniteElements.get_basis_on_cell(FE, ET),qf.xref[i])
        end    
    end
    return FEbasis_caller{typeof(FE),AssembleTypeCELL}(FE,ncomponents,ndofs4cell,refbasisvals,refgradients,refgradients_2ndorder,coefficients,Ahandler,Piolahandler,A,Piola,0.0,xdim,offsets,offsets2,0,current_dofs,with_standard_derivs, with_higher_derivs)
end    

function FEbasis_caller_face(FE::AbstractH1FiniteElement, qf::QuadratureFormula)
    ETF = Grid.get_face_elemtype(FE.grid.elemtypes[1])
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    refbasisvals = Array{Array{Float64,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        if ncomponents == 1
            refbasisvals[i] = reshape(FiniteElements.get_basis_on_face(FE, ETF)(qf.xref[i]),:,1)
        else
            refbasisvals[i] = FiniteElements.get_basis_on_face(FE, ETF)(qf.xref[i])
        end    
    end    
    ndofs4face = size(refbasisvals[1],1)
    coefficients = zeros(Float64,ndofs4face,ncomponents)
    Ahandler = Grid.local2global_tinv_jacobian(FE.grid,ETF)
    Piolahandler = (x) -> 0; #dummy function
    xdim = size(FE.grid.coords4nodes,2)
    A = zeros(Float64,xdim,xdim)
    Piola = zeros(Float64,0,0)
    offsets = 0:xdim:(ncomponents*xdim);
    offsets2 = 0:ndofs4face:ncomponents*ndofs4face;
    refgradients = zeros(Float64,0,0,0)
    refgradients_2ndorder = zeros(Float64,0,0,0)
    current_dofs = zeros(Int64,ndofs4face)
    return FEbasis_caller{typeof(FE),AssembleTypeFACE}(FE,ncomponents,ndofs4face,refbasisvals,refgradients,refgradients_2ndorder,coefficients,Ahandler,Piolahandler,A,Piola,0.0,xdim,offsets,offsets2,0,current_dofs,false,false)
end  

function FEbasis_caller_face(FE::AbstractHdivFiniteElement, qf::QuadratureFormula)
    ETF = Grid.get_face_elemtype(FE.grid.elemtypes[1])
    
    # pre-allocate memory for basis functions
    ncomponents = 1; # normal-fluxes are scalar
    refbasisvals = Array{Array{Float64,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        if ncomponents == 1
            refbasisvals[i] = reshape(FiniteElements.get_basis_fluxes_on_face(FE, ETF)(qf.xref[i]),:,1)
        else
            refbasisvals[i] = FiniteElements.get_basis_fluxes_on_face(FE, ETF)(qf.xref[i])
        end    
    end    
    ndofs4face = size(refbasisvals[1],1)
    coefficients = zeros(Float64,ndofs4face,ncomponents)
    Ahandler = (x) -> 0; #dummy function
    Piolahandler = (x) -> 0; #dummy function
    xdim = size(FE.grid.coords4nodes,2)
    A = zeros(Float64,0,0)
    Piola = zeros(Float64,0,0)
    offsets = [0];
    offsets2 = [0];
    refgradients = zeros(Float64,0,0,0)
    refgradients_2ndorder = zeros(Float64,0,0,0)
    current_dofs = zeros(Int64,ndofs4face)
    return FEbasis_caller{typeof(FE),AssembleTypeFACE}(FE,ncomponents,ndofs4face,refbasisvals,refgradients,refgradients_2ndorder,coefficients,Ahandler,Piolahandler,A,Piola,0.0,xdim,offsets,offsets2,0,current_dofs,false,false)
end 

function updateFEbasis!(FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractH1FiniteElement, cell)
    current_item = cell;

    # get coefficients
    FiniteElements.get_basis_coefficients_on_cell!(FEBC.coefficients, FEBC.FE, cell, FEBC.FE.grid.elemtypes[1]);

    # get dofs
    FiniteElements.get_dofs_on_cell!(FEBC.current_dofs,FEBC.FE,cell,FEBC.FE.grid.elemtypes[1]);

    # evaluate tinverted (=transposed + inverted) jacobian of element trafo
    if FEBC.with_standard_derivs || FEBC.with_higher_derivs
        FEBC.Ahandler(FEBC.A, cell)
    end    
end


function updateFEbasis!(FEBC::FEbasis_caller{FET,AssembleTypeFACE} where  FET <: AbstractH1FiniteElement, face)
    current_item = face;

    # get coefficients
    FiniteElements.get_basis_coefficients_on_face!(FEBC.coefficients, FEBC.FE, face, Grid.get_face_elemtype(FEBC.FE.grid.elemtypes[1]));
    
    # get dofs
    FiniteElements.get_dofs_on_face!(FEBC.current_dofs,FEBC.FE,face,Grid.get_face_elemtype(FEBC.FE.grid.elemtypes[1]));

end


function getFEbasis4qp!(basisvals,FEBC::FEbasis_caller{FET,AT} where {FET <: AbstractH1FiniteElement, AT <: AbstractAssembleType},i)
    for j = 1 : FEBC.ndofs4item, k = 1 : FEBC.ncomponents
        basisvals[j,k] = FEBC.refbasisvals[i][j,k] * FEBC.coefficients[j,k]
    end    
end


function getFEbasisgradients4qp!(gradients,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractH1FiniteElement,i)
    @assert FEBC.with_standard_derivs
    # multiply tinverted jacobian of element trafo with gradient of basis function
    # which yields (by chain rule) the gradient in x coordinates
    for dof_i = 1 : FEBC.ndofs4item
        for c = 1 : FEBC.ncomponents, k = 1 : FEBC.xdim
            gradients[dof_i,k + FEBC.offsets[c]] = 0.0;
            for j = 1 : FEBC.xdim
                # compute duc/dxk
                gradients[dof_i,k + FEBC.offsets[c]] += FEBC.A[k,j]*FEBC.refgradients[i,dof_i + FEBC.offsets2[c],j]
            end    
            gradients[dof_i,k + FEBC.offsets[c]] *= FEBC.coefficients[dof_i,c]
        end    
    end    
end


# symmetric gradients are saved in reduced Voigt notation
# the following mapping tells where each entry of the full gradient lands in the reduced vector
voigt_mapper = Array{Array{Int64,1},1}(undef,3)
voigt_mapper[1] = [1]
voigt_mapper[2] = [1,3,3,2]
voigt_mapper[3] = [1,4,5,4,2,6,5,6,3]

function getFEbasissymgradients4qp!(symgradients,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractH1FiniteElement,i)
    @assert FEBC.with_standard_derivs
    # multiply tinverted jacobian of element trafo with gradient of basis function
    # which yields (by chain rule) the gradient in x coordinates
    for dof_i = 1 : FEBC.ndofs4item
        for k = 1 : size(symgradients,2)
            symgradients[dof_i,k] = 0.0;
        end
        for c = 1 : FEBC.ncomponents, k = 1 : FEBC.xdim
            for j = 1 : FEBC.xdim
                # compute duc/dxk
                symgradients[dof_i,voigt_mapper[FEBC.ncomponents][k + FEBC.offsets[c]]] += FEBC.A[k,j]*FEBC.refgradients[i,dof_i + FEBC.offsets2[c],j]*FEBC.coefficients[dof_i,c]
            end    
        end    
    end    
end

function getFEbasiscurls4qp!(curls,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractH1FiniteElement,i)
    @assert FEBC.with_standard_derivs
    @assert FEBC.xdim == 2 # 3D curl not yet implemented
    @assert FEBC.xdim == FEBC.ncomponents # only defined for vector-fields

    # in 2D : curl = du2/dx - du1/dy
    for dof_i = 1 : FEBC.ndofs4item
        curls[dof_i,1] = 0.0;
        for j = 1 : FEBC.xdim
            # du2/dx
            curls[dof_i,1] += FEBC.A[1,j]*FEBC.refgradients[i,dof_i + FEBC.offsets2[2],j]
            # -du1/dy
            curls[dof_i,1] -= FEBC.A[2,j]*FEBC.refgradients[i,dof_i + FEBC.offsets2[1],j]
        end
        curls[dof_i,1] *= FEBC.coefficients[dof_i,1]
    end    
end

function getFEbasislaplacians4qp!(laplacians,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractH1FiniteElement,i,diffusion_matrix)
    @assert FEBC.with_higher_derivs
    for dof_i = 1 : FEBC.ndofs4item
        for c = 1 : FEBC.ncomponents
            laplacians[dof_i,c] = 0.0;
            for xi = 1 : FEBC.xdim, xj = 1: FEBC.xdim
                # add second derivatives diffusion[j,k]*partial^2 (x_i x_j)
                if diffusion_matrix[xi,xj] != 0
                    for k = 1 : FEBC.xdim, l = 1 : FEBC.xdim
                        laplacians[dof_i,c] += diffusion_matrix[xi,xj]*FEBC.A[xi,k]*FEBC.A[xj,l]*FEBC.refgradients_2ndorder[i,dof_i + FEBC.offsets2[k],l]
                    end
                end    
            end    
            laplacians[dof_i,c] *= FEBC.coefficients[dof_i,c]
        end    
    end    
end


function FEbasis_caller(FE::AbstractHdivFiniteElement, qf::QuadratureFormula, with_divergence::Bool, with_other_derivatives::Bool = false)
    ET = FE.grid.elemtypes[1]
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    refbasisvals = Array{Array{Float64,2},1}(undef,length(qf.w));
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        if ncomponents == 1
            refbasisvals[i] = reshape(FiniteElements.get_basis_on_cell(FE, ET)(qf.xref[i]),:,1)
        else
            refbasisvals[i] = FiniteElements.get_basis_on_cell(FE, ET)(qf.xref[i])
        end    
    end    
    ndofs4cell = size(refbasisvals[1],1)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    Ahandler = Grid.local2global_tinv_jacobian(FE.grid,ET)
    xdim = size(FE.grid.coords4nodes,2)
    A = zeros(Float64,xdim,xdim)
    Piola = zeros(Float64,xdim,xdim)
    Piolahandler = Grid.local2global_Piola(FE.grid, ET)
    offsets = 0:xdim:(ncomponents*xdim);
    offsets2 = 0:ndofs4cell:ncomponents*ndofs4cell;
    if (with_divergence == false) && (with_other_derivatives == false)
        refgradients = zeros(Float64,0,0,0)
    else
        refgradients = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,length(qf.xref[1]))
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refgradients[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_cell(FE, ET),qf.xref[i]);
        end    
    end
    refgradients_2ndorder = zeros(Float64,0,0,0)
    current_dofs = zeros(Int64,ndofs4cell)
    return FEbasis_caller{typeof(FE),AssembleTypeCELL}(FE,ncomponents,ndofs4cell,refbasisvals,refgradients,refgradients_2ndorder,coefficients,Ahandler,Piolahandler,A,Piola,0.0,xdim,offsets,offsets2,0,current_dofs,with_divergence,with_other_derivatives)
end    

function updateFEbasis!(FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractHdivFiniteElement, cell)
    current_item = cell;

    # get Piola trafo
    FEBC.det = FEBC.Piolahandler(FEBC.Piola,cell);

    # get coefficients
    FiniteElements.get_basis_coefficients_on_cell!(FEBC.coefficients, FEBC.FE, cell, FEBC.FE.grid.elemtypes[1]);

    # get dofs
    FiniteElements.get_dofs_on_cell!(FEBC.current_dofs,FEBC.FE,cell,FEBC.FE.grid.elemtypes[1]);

    if FEBC.with_higher_derivs # todo: not needed if only divergence of Hdiv function is computed!
        FEBC.Ahandler(FEBC. A, cell)
    end    
end


function updateFEbasis!(FEBC::FEbasis_caller{FET,AssembleTypeFACE} where  FET <: AbstractHdivFiniteElement, face)
    current_item = face;

    # get Piola trafo
    FEBC.det = FEBC.FE.grid.length4faces[face];

    # get coefficients
    FiniteElements.get_basis_coefficients_on_face!(FEBC.coefficients, FEBC.FE, face, FEBC.FE.grid.elemtypes[1]);

    # get dofs
    FiniteElements.get_dofs_on_face!(FEBC.current_dofs,FEBC.FE,face,Grid.get_face_elemtype(FEBC.FE.grid.elemtypes[1]));

end

function getFEbasis4qp!(basisvals,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractHdivFiniteElement,i)
    # use Piola transformation on basisvals
    for j = 1 : FEBC.ndofs4item
        for k = 1 : FEBC.xdim
            basisvals[j,k] = 0.0;
            for l = 1 : FEBC.xdim
                basisvals[j,k] += FEBC.Piola[k,l]*FEBC.refbasisvals[i][j,l];
            end    
            basisvals[j,k] *= FEBC.coefficients[j,k];
            basisvals[j,k] /= FEBC.det;
        end
    end   
end


function getFEbasis4qp!(basisvals,FEBC::FEbasis_caller{FET,AssembleTypeFACE} where  FET <: AbstractHdivFiniteElement,i)
    for j = 1 : FEBC.ndofs4item # only normal-fluxes
        basisvals[j,1] = FEBC.refbasisvals[i][j,1]
        basisvals[j,1] *= FEBC.coefficients[j,1];
        basisvals[j,1] /= FEBC.det;
    end    
end

function getFEbasisdivergence4qp!(divergences,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractH1FiniteElement,i)
    # Piola transformation preserves divergence (up to a factor 1/det(A))
    for dof_i = 1 : FEBC.ndofs4item
        divergences[dof_i,1] = 0.0;
        for k = 1 : FEBC.xdim
            for j = 1 : FEBC.xdim
                divergences[dof_i,1] += FEBC.A[k,j] * FEBC.refgradients[i,dof_i + FEBC.offsets2[k],j] * FEBC.coefficients[dof_i,k]
            end    
        end    
    end    
end


function getFEbasisdivergence4qp!(divergences,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractHdivFiniteElement,i)
    # Piola transformation preserves divergence (up to a factor 1/det(A))
    for dof_i = 1 : FEBC.ndofs4item
        divergences[dof_i,1] = 0.0;
        for j = 1 : FEBC.xdim
            divergences[dof_i,1] += FEBC.refgradients[i,dof_i + FEBC.offsets2[j],j]
        end  
        divergences[dof_i,1] *= FEBC.coefficients[dof_i,1]/FEBC.det;
    end    
end



function getFEbasiscurls4qp!(curls,FEBC::FEbasis_caller{FET,AssembleTypeCELL} where  FET <: AbstractHdivFiniteElement,i)
    @assert FEBC.with_higher_derivs
    @assert FEBC.xdim == 2 # 3D curl not yet implemented
    @assert FEBC.xdim == FEBC.ncomponents # only defined for vector-fields

    # in 2D : curl = du2/dx - du1/dy
    for dof_i = 1 : FEBC.ndofs4item
        curls[dof_i,1] = 0.0;
        for j = 1 : FEBC.xdim, k = 1 : FEBC.xdim
            # du2/dx
            curls[dof_i,1] += FEBC.A[1,j]*FEBC.Piola[2,k]*FEBC.refgradients[i,dof_i + FEBC.offsets2[k],j]
            # -du1/dy
            curls[dof_i,1] -= FEBC.A[2,j]*FEBC.Piola[1,k]*FEBC.refgradients[i,dof_i + FEBC.offsets2[k],j]
        end
        curls[dof_i,1] *= FEBC.coefficients[dof_i,1]    
    end    
end


end #module
