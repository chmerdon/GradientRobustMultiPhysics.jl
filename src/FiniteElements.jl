module FiniteElements

using Grid
using LinearAlgebra
using Quadrature
using ForwardDiff

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

# top level abstract type
abstract type AbstractFiniteElement end

  # subtype for H1 conforming elements (also Crouzeix-Raviart)
  abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/H1_P1.jl");
    include("FEdefinitions/H1_MINI.jl");
    include("FEdefinitions/H1_P2.jl");
    include("FEdefinitions/H1_CR.jl");
    include("FEdefinitions/H1_BR.jl");
 
  # subtype for L2 conforming elements
  abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/L2_P0.jl");
 
  # subtype for Hdiv-conforming elements
  abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/HDIV_RT0.jl");
    include("FEdefinitions/HDIV_RT1.jl");

  # subtype for Hcurl-conforming elements
  abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end
    # TODO


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

# creates a zero vector of the correct length for the FE
function createFEVector(FE::AbstractFiniteElement)
    return zeros(get_ndofs(FE));
end

mutable struct FEbasis_caller{FEType <: AbstractFiniteElement}
    FE::AbstractFiniteElement
    refbasisvals::Array{Float64,3}
    refgradients::Array{Float64,3}
    coefficients::Array{Float64,2}
    Ahandler::Function  # function that generates matrix trafo
    A::Matrix{Float64}  # matrix trafo for gradients
    det::Float64        # determinant of matrix
    ncomponents::Int64
    offsets::Array{Int64,1}
    offsets2::Array{Int64,1}
    current_cell::Int64
    with_derivs::Bool
end

function FEbasis_caller(FE::AbstractH1FiniteElement, qf::QuadratureFormula, with_derivs::Bool)
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    refbasisvals = zeros(Float64,length(qf.w),ndofs4cell,ncomponents);
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        refbasisvals[i,:,:] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    Ahandler = Grid.local2global_tinv_jacobian(FE.grid,ET)
    xdim = size(FE.grid.coords4nodes,2)
    A = zeros(Float64,xdim,xdim)
    offsets = 0:xdim:(ncomponents*xdim);
    offsets2 = 0:ndofs4cell:ncomponents*ndofs4cell;
    if with_derivs == false
        refgradients = zeros(Float64,0,0,0)
    else
        refgradients = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,length(qf.xref[1]))
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refgradients[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE, ET),qf.xref[i]);
        end    
    end
    return FEbasis_caller{typeof(FE)}(FE,refbasisvals,refgradients,coefficients,Ahandler,A,0.0,ncomponents,offsets,offsets2,0,with_derivs)
    
end    

function updateFEbasis!(FEBC::FEbasis_caller{FET} where  FET <: AbstractH1FiniteElement, cell)
    current_cell = cell;

    # get coefficients
    FiniteElements.get_basis_coefficients_on_cell!(FEBC.coefficients, FEBC.FE, cell, FEBC.FE.grid.elemtypes[1]);

    # evaluate tinverted (=transposed + inverted) jacobian of element trafo
    if FEBC.with_derivs
        FEBC.Ahandler(FEBC.A, cell)
    end    
end

function getFEbasis4qp!(basisvals,FEBC::FEbasis_caller{FET} where  FET <: AbstractH1FiniteElement,i)
    for j = 1 : size(FEBC.refbasisvals,2), k = 1 : size(FEBC.refbasisvals,3)
        basisvals[j,k] = FEBC.refbasisvals[i,j,k] * FEBC.coefficients[j,k]
    end    
end


function getFEbasisgradients4qp!(gradients,FEBC::FEbasis_caller{FET} where  FET <: AbstractH1FiniteElement,i)
    @assert FEBC.with_derivs
    # multiply tinverted jacobian of element trafo with gradient of basis function
    # which yields (by chain rule) the gradient in x coordinates
    for dof_i = 1 : size(FEBC.refbasisvals,2)
        for c = 1 : FiniteElements.get_ncomponents(FEBC.FE), k = 1 : size(FEBC.FE.grid.coords4nodes,2)
            gradients[dof_i,k + FEBC.offsets[c]] = 0.0;
            for j = 1 : size(FEBC.FE.grid.coords4nodes,2)
                gradients[dof_i,k + FEBC.offsets[c]] += FEBC.A[k,j]*FEBC.refgradients[i,dof_i + FEBC.offsets2[c],j] * FEBC.coefficients[dof_i,c]
            end    
        end    
    end    
end


function FEbasis_caller(FE::AbstractHdivFiniteElement, qf::QuadratureFormula, with_derivs::Bool)
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    refbasisvals = zeros(Float64,length(qf.w),ndofs4cell,ncomponents);
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        refbasisvals[i,:,:] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    Ahandler = Grid.local2global_Piola(FE.grid, ET)
    xdim = size(FE.grid.coords4nodes,2)
    A = zeros(Float64,xdim,xdim)
    offsets = 0:xdim:(ncomponents*xdim);
    offsets2 = 0:ndofs4cell:ncomponents*ndofs4cell;
    if with_derivs == false
        refgradients = zeros(Float64,0,0,0)
    else
        refgradients = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,length(qf.xref[1]))
        for i in eachindex(qf.w)
            # evaluate gradients of basis function
            refgradients[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE, ET),qf.xref[i]);
        end    
    end
    return FEbasis_caller{typeof(FE)}(FE,refbasisvals,refgradients,coefficients,Ahandler,A,0.0,ncomponents,offsets,offsets2,0,with_derivs)
end    

function updateFEbasis!(FEBC::FEbasis_caller{FET} where  FET <: AbstractHdivFiniteElement, cell)
    current_cell = cell;

    # get Piola trafo
    FEBC.det = FEBC.Ahandler(FEBC.A,cell);

    # get coefficients
    FiniteElements.get_basis_coefficients_on_cell!(FEBC.coefficients, FEBC.FE, cell, FEBC.FE.grid.elemtypes[1]);
end

function getFEbasis4qp!(basisvals,FEBC::FEbasis_caller{FET} where  FET <: AbstractHdivFiniteElement,i)
    # use Piola transformation on basisvals
    for j = 1 : size(FEBC.refbasisvals,2)
        for k = 1 : size(FEBC.refbasisvals,3)
            basisvals[j,k] = 0.0;
            for l = 1 : size(FEBC.refbasisvals,3)
                basisvals[j,k] += FEBC.A[k,l]*FEBC.refbasisvals[i,j,l];
            end    
            basisvals[j,k] *= FEBC.coefficients[j,k];
            basisvals[j,k] /= FEBC.det;
        end
    end   
end


end #module
