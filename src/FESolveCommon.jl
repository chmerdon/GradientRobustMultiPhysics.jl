module FESolveCommon

export accumarray, computeBestApproximation!, computeFEInterpolation!, eval_interpolation_error!, eval_L2_interpolation_error!

using SparseArrays
using ExtendableSparse
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using DiffResults
using ForwardDiff
using Grid
using Quadrature

function accumarray!(A,subs, val, sz=(maximum(subs),))
    for i = 1:length(val)
        A[subs[i]] += val[i]
    end
end



# matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(FE.grid.elemtypes[1])}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents > 1
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    end    
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
                    
                
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,xdim)
    
    # quadrature loop
    temp = 0.0;
    #@time begin    
    for cell = 1 : ncells
        
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

        # get coefficients
        FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell);
            
        for i in eachindex(qf.w)
            for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                # fill upper right part and diagonal of matrix
                @inbounds begin
                    temp = 0.0
                    for k = 1 : ncomponents
                        temp += (basisvals[i][dof_i,k]*basisvals[i][dof_j,k] * qf.w[i] * FE.grid.volume4cells[cell]) * coefficients[dof_i,k] * coefficients[dof_j,k];
                    end
                    A[dofs[dof_i],dofs[dof_j]] += temp;
                    # fill lower left part of matrix
                    if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                    end    
                end
            end
        end
    end
    #end    
end



# matrix for L2 bestapproximation that writes into an ExtendableSparseMatrix
function assemble_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractHdivFiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ET)}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents > 1
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    end    
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
    transformed_basisvals = zeros(ndofs4cell,ncomponents);
                    
                
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,xdim)

    AT = zeros(Float64,2,2);
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, ET)
    
    # quadrature loop
    temp = 0.0;
    det = 0.0;
    #@time begin    
        for cell = 1 : ncells
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

            # get Piola trafo
            det = get_Piola_trafo_on_cell!(AT,cell);
            
            # get coefficients
            FiniteElements.set_basis_coefficients_on_cell!(coefficients,FE,cell);
            
            for i in eachindex(qf.w)
                # use Piola transformation on basisvals
                for dof_i = 1 : ndofs4cell
                    for k = 1 : ncomponents
                        transformed_basisvals[dof_i,k] = 0.0;
                        for l = 1 : ncomponents
                            transformed_basisvals[dof_i,k] += AT[k,l]*basisvals[i][dof_i,l];
                        end    
                        transformed_basisvals[dof_i,k] /= det;
                    end    
                end    
                for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                    # fill upper right part and diagonal of matrix
                    @inbounds begin

                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (transformed_basisvals[dof_i,k]*transformed_basisvals[dof_j,k] * qf.w[i] * FE.grid.volume4cells[cell]) * coefficients[dof_i,k] * coefficients[dof_j,k];
                      end
                      A[dofs[dof_i],dofs[dof_j]] += temp;
                      # fill lower left part of matrix
                      if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                      end    
                    end
                end
            end
        end#
    #end   
end

# matrix for L2 bestapproximation on boundary faces that writes into an ExtendableSparseMatrix
function assemble_bface_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractH1FiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,Grid.ElemType1DInterval}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents > 1
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    end    
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face,xdim)
    
    # quadrature loop
    temp = 0.0;
    face = 0;
    #@time begin    
        for j in eachindex(FE.grid.bfaces)
            face = FE.grid.bfaces[j];
            
            # get dofs
            FiniteElements.get_dofs_on_face!(dofs,FE,face,ETF);

            # get coefficients
            FiniteElements.set_basis_coefficients_on_face!(coefficients,FE,face);
            
            for i in eachindex(qf.w)
                
                for dof_i = 1 : ndofs4face, dof_j = dof_i : ndofs4face
                    # fill upper right part and diagonal of matrix
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (basisvals[i][dof_i,k]*basisvals[i][dof_j,k] * qf.w[i] * FE.grid.length4faces[face]) * coefficients[dof_i,k] * coefficients[dof_j,k];
                      end
                      A[dofs[dof_i],dofs[dof_j]] += temp;
                      # fill lower left part of matrix
                      if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                      end 
                    end
                end
            end
        end#
    #end      
end


function assemble_bface_mass_matrix4FE!(A::ExtendableSparseMatrix,FE::AbstractHdivFiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET)
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ETF)}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = Array{Array{T,1}}(undef,length(qf.w));
        
    for i in eachindex(qf.w)
        # evaluate basis functions at quadrature point
        basisvals[i] = FiniteElements.get_basis_fluxes_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face)
    
    # quadrature loop
    det = 0.0;
    temp = 0.0;
    face = 0;
    #@time begin    
        for j in eachindex(FE.grid.bfaces)
            face = FE.grid.bfaces[j];

            # get dofs
            FiniteElements.get_dofs_on_face!(dofs, FE, face, ETF);

            # get coefficients
            FiniteElements.set_basis_coefficients_on_face!(coefficients,FE,face);

            det = FE.grid.length4faces[face]; # determinant of transformation on face
            
            for i in eachindex(qf.w)
                
                for dof_i = 1 : ndofs4face, dof_j = dof_i : ndofs4face
                    # fill upper right part and diagonal of matrix
                    @inbounds begin
                      temp = (basisvals[i][dof_i]*basisvals[i][dof_j] * qf.w[i] * FE.grid.length4faces[face]) * coefficients[dof_i] * coefficients[dof_j] / det / det;
                      A[dofs[dof_i],dofs[dof_j]] += temp;
                      # fill lower left part of matrix
                      if dof_j > dof_i
                        A[dofs[dof_j],dofs[dof_i]] += temp;
                      end 
                    end
                end
            end
        end#
    #end      
end



# stiffness matrix assembly that writes into an ExtendableSparseMatrix
function assemble_stiffness_matrix4FE!(A::ExtendableSparseMatrix,nu::Real,FE::AbstractH1FiniteElement)
    ncells::Int = size(FE.grid.nodes4cells,1);
    T = eltype(FE.grid.coords4nodes);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    ncomponents::Int = FiniteElements.get_ncomponents(FE);
    
    qf = QuadratureFormula{T,typeof(ET)}(2*(FiniteElements.get_polynomial_order(FE)-1));
    
    # pre-allocate for derivatives of global2local trafo and basis function
    gradients_xref_cache = zeros(Float64,length(qf.w),ncomponents*ndofs4cell,celldim)
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,ndofs4cell,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dofs = zeros(Int64,ndofs4cell)
    gradients4cell = Array{Array{T,1}}(undef,ncomponents*ndofs4cell);
    coefficients = zeros(Float64,ndofs4cell,xdim)
    for j = 1 : ncomponents*ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE, ET),qf.xref[i]);
    end    

    loc2glob_trafo_tinv = Grid.local2global_tinv_jacobian(FE.grid,ET)
          
    # quadrature loop
    temp::T = 0.0;
    offsets = [0,ndofs4cell];
    #@time begin
    for cell = 1 : ncells
      
      # evaluate tinverted (=transposed + inverted) jacobian of element trafo
      loc2glob_trafo_tinv(trafo_jacobian,cell)

      # get dofs
      FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

      # get coefficients
      FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell);
      
      for i in eachindex(qf.w)
        
        # multiply tinverted jacobian of element trafo with gradient of basis function
        # which yields (by chain rule) the gradient in x coordinates
        for dof_i = 1 : ndofs4cell
            for c=1 : ncomponents, k = 1 : xdim
                gradients4cell[dof_i + offsets[c]][k] = 0.0;
                for j = 1 : xdim
                    gradients4cell[dof_i + offsets[c]][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i + offsets[c],j] * coefficients[dof_i,c]
                end    
            end    
        end    
        
        # fill sparse array
        for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
            # fill upper right part and diagonal of matrix
            temp = 0.0;
            for k = 1 : xdim
                for c = 1 : ncomponents
                    temp += gradients4cell[offsets[c]+dof_i][k]*gradients4cell[offsets[c]+dof_j][k];
                end
            end
            temp *= nu * qf.w[i] * FE.grid.volume4cells[cell]
            A[dofs[dof_i],dofs[dof_j]] += temp;
            # fill lower left part of matrix
            if dof_j > dof_i
              A[dofs[dof_j],dofs[dof_i]] += temp;
            end    
          end
      end  
    end
    #end
end


function assemble_rhsL2!(b, f!::Function, FE::AbstractH1FiniteElement, quadrature_order::Int64)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(FE.grid.elemtypes[1])}(quadrature_order + FiniteElements.get_polynomial_order(FE));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    end
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,xdim)
    
    loc2glob_trafo = Grid.local2global(FE.grid,FE.grid.elemtypes[1])
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,ncomponents)
    #@time begin    
        for cell = 1 : ncells

            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);
            
            # get coefficients
            FiniteElements.set_basis_coefficients_on_cell!(coefficients,FE,cell);

            # setup trafo on cell
            cell_trafo = loc2glob_trafo(cell)
            
            for i in eachindex(qf.w)
                
                # evaluate f
                x = cell_trafo(qf.xref[i]);
                f!(fval, x)
                
                for dof_i = 1 : ndofs4cell
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (fval[k]*basisvals[i][dof_i,k]*coefficients[dof_i,k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
    #end    
end

function assemble_rhsL2!(b, f!::Function, FE::AbstractHdivFiniteElement, quadrature_order::Int64)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ET)}(quadrature_order + FiniteElements.get_polynomial_order(FE));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    end
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(qf.xref[i])
    end    
    transformed_basisvals = zeros(ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,xdim)
    
    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    
    AT = zeros(Float64,2,2)
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, FE.grid.elemtypes[1])
    
    # quadrature loop
    temp = 0.0;
    det = 0.0;
    fval = zeros(T,ncomponents)
   # @time begin    
        for cell = 1 : ncells
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);
            
            # get coefficients
            FiniteElements.set_basis_coefficients_on_cell!(coefficients,FE,cell);
            
            # get trafo and Piola trafo
            cell_trafo = loc2glob_trafo(cell)
            det = get_Piola_trafo_on_cell!(AT,cell);
            
            for i in eachindex(qf.w)
                
                # use Piola transformation on basisvals
                for dof_i = 1 : ndofs4cell
                    for k = 1 : ncomponents
                        transformed_basisvals[dof_i,k] = 0.0;
                        for l = 1 : ncomponents
                            transformed_basisvals[dof_i,k] += AT[k,l]*basisvals[i][dof_i,l];
                        end    
                        transformed_basisvals[dof_i,k] /= det;
                    end    
                end    

                # evaluate f
                x = cell_trafo(qf.xref[i]);
                f!(fval, x)
                
                for dof_i = 1 : ndofs4cell
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (fval[k]*transformed_basisvals[dof_i,k]*coefficients[dof_i,k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
   # end    
end

# TODO: use ElemTypes
function assemble_rhsL2_on_bface!(b, f!::Function, FE::AbstractH1FiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,Grid.ElemType1DInterval}(2*(FiniteElements.get_polynomial_order(FE)));
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,Grid.ElemType1DInterval}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,length(qf.w));
    else
        basisvals = Array{Array{T,2}}(undef,length(qf.w));
    end
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face,xdim)
    
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)

    temp = 0.0
    fval = zeros(T,ncomponents)
    x = zeros(T,xdim);
    face = 0
    @time begin    
        for j in eachindex(FE.grid.bfaces)

            face = FE.grid.bfaces[j];
            
            # get dofs
            FiniteElements.get_dofs_on_face!(dofs,FE,face,ETF);

            # get coefficients
            FiniteElements.set_basis_coefficients_on_face!(coefficients,FE,face);

            # get face trafo
            face_trafo = loc2glob_trafo(face)
            
            for i in eachindex(qf.w)
                # evaluate f
                x = face_trafo(qf.xref[i])
                f!(fval, x)
                
                for dof_i = 1 : ndofs4face
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : ncomponents
                        temp += (fval[k]*basisvals[i][dof_i,k] * qf.w[i] * FE.grid.length4faces[face] * coefficients[dof_i,k]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
    end    
end

function assemble_rhsL2_on_bface!(b, f!::Function, FE::AbstractHdivFiniteElement)
    ensure_bfaces!(FE.grid);
    ensure_length4faces!(FE.grid);
    nbfaces::Int = size(FE.grid.bfaces,1);
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET)
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
    celldim::Int = size(FE.grid.nodes4cells,2) - 1;
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ETF)}(2*(FiniteElements.get_polynomial_order(FE)));
     
    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    basisvals = Array{Array{T,1}}(undef,length(qf.w));
    for i in eachindex(qf.w)
        basisvals[i] = FiniteElements.get_basis_fluxes_on_elemtype(FE, ETF)(qf.xref[i])
    end    
    dofs = zeros(Int64,ndofs4face)
    coefficients = zeros(Float64,ndofs4face)
    
    loc2glob_trafo = Grid.local2global_face(FE.grid, ET)
    
    # quadrature loop
    det = 0.0
    fval = zeros(T,ncomponents)
    x = zeros(T,xdim);
    face = 0
    @time begin    
        for j in eachindex(FE.grid.bfaces)
            face = FE.grid.bfaces[j];

            # get dofs
            FiniteElements.get_dofs_on_face!(dofs, FE, face, ETF);
            
            # get coefficients
            FiniteElements.set_basis_coefficients_on_face!(coefficients,FE,face);

            # get face trafo
            face_trafo = loc2glob_trafo(face)

            # get Piola trafo (todo: use elemtype-steered trafo)
            det = FE.grid.length4faces[face]; # determinant of transformation on face
            
            for i in eachindex(qf.w)
                # evaluate f
                x = face_trafo(qf.xref[i])
                f!(fval, x)

                # multiply with normal and save in fval[1]
                fval[1] = fval[1] * FE.grid.normal4faces[face,1] + fval[2] * FE.grid.normal4faces[face,2];
                fval[2] = 0.0;

                for dof_i = 1 : ndofs4face
                    # fill vector
                    # note: basisvals contain normalfluxes that have to be scaled with 1/det (Piola)
                    @inbounds begin
                      b[dofs[dof_i]] += (fval[1]*basisvals[i][dof_i,1]/det * qf.w[i] * FE.grid.length4faces[face] * coefficients[dof_i]);;
                    end
                end
            end
        end
    end    
end



function assemble_rhsH1!(b, f!::Function, FE::AbstractH1FiniteElement, quadrature_order::Int)
    ncells::Int = size(FE.grid.nodes4cells,1);
    ET = FE.grid.elemtypes[1];
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    xdim::Int = size(FE.grid.coords4nodes,2);
    celldim::Int = size(FE.grid.nodes4cells,2);
    
    T = eltype(FE.grid.coords4nodes);
    qf = QuadratureFormula{T,typeof(ET)}(quadrature_order + min(0,FiniteElements.get_polynomial_order(FE) - 1));
     
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,ndofs4cell);
    for j = 1: ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    gradients_xref_cache = zeros(Float64,length(qf.w),ndofs4cell,celldim)
    for i in eachindex(qf.w)
        # evaluate gradients of basis function
        gradients_xref_cache[i,:,:] = ForwardDiff.jacobian(FiniteElements.get_basis_on_elemtype(FE, ET),qf.xref[i]);
    end    
    #DRresult_grad = DiffResults.DiffResult(Vector{T}(undef, celldim), Matrix{T}(undef,ndofs4cell,celldim));
    trafo_jacobian = Matrix{T}(undef,xdim,xdim);
    dofs = zeros(Int64,ndofs4cell)
    
    dim = celldim - 1;
    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    loc2glob_trafo_tinv = Grid.local2global_tinv_jacobian(FE.grid,ET)
    
    
    # quadrature loop
    temp = 0.0;
    fval = zeros(T,xdim)
    @time begin    
        for cell = 1 : ncells
      
            # evaluate tinverted (=transposed + inverted) jacobian of element trafo
            loc2glob_trafo_tinv(trafo_jacobian,cell)
      
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);      

            # get trafo
            cell_trafo = loc2glob_trafo(cell)
      
            for i in eachindex(qf.w)
        
                # multiply tinverted jacobian of element trafo with gradient of basis function
                # which yields (by chain rule) the gradient in x coordinates
                for dof_i = 1 : ndofs4cell
                    for k = 1 : xdim
                        gradients4cell[dof_i][k] = 0.0;
                        for j = 1 : xdim
                            gradients4cell[dof_i][k] += trafo_jacobian[k,j]*gradients_xref_cache[i,dof_i,j]
                        end    
                    end    
                end 
                
                # evaluate f
                x = cell_trafo(qf.xref[i]);
                f!(fval, x)
                
                for dof_i = 1 : ndofs4cell
                    # fill vector
                    @inbounds begin
                      temp = 0.0
                      for k = 1 : xdim
                        temp += (fval[k] * gradients4cell[dof_i][k] * qf.w[i] * FE.grid.volume4cells[cell]);
                      end
                      b[dofs[dof_i]] += temp;
                    end
                end
            end
        end
    end    
end


function assembleSystem(nu::Real, norm_lhs::String,norm_rhs::String,volume_data!::Function,FE::AbstractFiniteElement,quadrature_order::Int)

    ncells::Int = size(FE.grid.nodes4cells,1);
    nnodes::Int = size(FE.grid.coords4nodes,1);
    celldim::Int = size(FE.grid.nodes4cells,2);
    xdim::Int = size(FE.grid.coords4nodes,2);
    
    Grid.ensure_volume4cells!(FE.grid);
    
    ndofs = FiniteElements.get_ndofs(FE);
    A = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs);
    if norm_lhs == "L2"
        assemble_mass_matrix4FE!(A,FE);
    elseif norm_lhs == "H1"
        assemble_stiffness_matrix4FE!(A,nu,FE);
    end 
    
    # compute right-hand side vector
    b = FiniteElements.createFEVector(FE);
    if norm_rhs == "L2"
        assemble_rhsL2!(b, volume_data!, FE, quadrature_order)
    elseif norm_rhs == "H1"
        assemble_rhsH1!(b, volume_data!, FE, quadrature_order)
    end
    
    
    return A,b
end

function computeDirichletBoundaryData!(val4dofs,FE,boundary_data!,use_L2bestapproximation = false)
    if (boundary_data! == Nothing)
        return []
    else
        if use_L2bestapproximation == false
            computeDirichletBoundaryDataByInterpolation!(val4dofs,FE,boundary_data!);
        else
            ndofs = FiniteElements.get_ndofs(FE);
            B = ExtendableSparseMatrix{Float64,Int64}(ndofs,ndofs)
            assemble_bface_mass_matrix4FE!(B::ExtendableSparseMatrix,FE)
            b = FiniteElements.createFEVector(FE);
            assemble_rhsL2_on_bface!(b, boundary_data!, FE)
        
            ETF = Grid.get_face_elemtype(FE.grid.elemtypes[1]);
            ensure_bfaces!(FE.grid);
            nbfaces::Int = size(FE.grid.bfaces,1);
            ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);
        
            bdofs = [];
            dofs = zeros(Int64,ndofs4face)
            for face in eachindex(FE.grid.bfaces)
                FiniteElements.get_dofs_on_face!(dofs, FE, FE.grid.bfaces[face], ETF);
                append!(bdofs,dofs)
            end
        
            unique!(bdofs)
            val4dofs[bdofs] = B[bdofs,bdofs]\b[bdofs];
            return bdofs
        end    
    end
end


function computeDirichletBoundaryDataByInterpolation!(val4dofs,FE,boundary_data!)

    # find boundary dofs
    xdim = FiniteElements.get_ncomponents(FE);
    ndofs::Int = FiniteElements.get_ndofs(FE);
    
    bdofs = [];
    Grid.ensure_bfaces!(FE.grid);
    Grid.ensure_cells4faces!(FE.grid);
    T = eltype(FE.grid.coords4nodes)
    ET = FE.grid.elemtypes[1]
    ETF = Grid.get_face_elemtype(ET);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    ndofs4face::Int = FiniteElements.get_ndofs4elemtype(FE, ETF);

    xref = FiniteElements.get_xref4dof(FE, ET)
    temp = zeros(eltype(FE.grid.coords4nodes),xdim);
    dim = size(FE.grid.nodes4cells,2) - 1;
    loc2glob_trafo = Grid.local2global(FE.grid, ET)

    # pre-allocate memory for basis functions
    ncomponents = FiniteElements.get_ncomponents(FE);
    if ncomponents == 1
        basisvals = Array{Array{T,1}}(undef,size(xref,1));
    else
        basisvals = Array{Array{T,2}}(undef,size(xref,1));
    end
    for i = 1:size(xref,1)
        basisvals[i] = FiniteElements.get_basis_on_elemtype(FE, ET)(xref[i,:])
    end    

    cell::Int = 0;
    j::Int = 1;
    A4bface = Matrix{Float64}(undef,ndofs4face,ndofs4face)
    b4bface = Vector{Float64}(undef,ndofs4face)
    dofs4bface = zeros(Int64,ndofs4face)
    dofs4cell = zeros(Int64,ndofs4cell)
    celldof2facedof = zeros(Int,ndofs4face)
    for i in eachindex(FE.grid.bfaces)

        # find neighbouring cell
        cell = FE.grid.cells4faces[FE.grid.bfaces[i],1];

        # get dofs
        FiniteElements.get_dofs_on_face!(dofs4bface, FE, FE.grid.bfaces[i], ETF);
        FiniteElements.get_dofs_on_cell!(dofs4cell, FE, cell, ET);

        # find position of face dofs in cell dofs
        for j=1:ndofs4cell, k = 1 : ndofs4face
            if dofs4cell[j] == dofs4bface[k]
                celldof2facedof[k] = j;
            end   
        end

        # append face dofs to bdofs
        append!(bdofs,dofs4bface);

        # setup trafo
        trafo_on_cell = loc2glob_trafo(cell);

        # setup local system of equations to determine piecewise interpolation of boundary data

        # assemble matrix and right-hand side
        for k = 1:ndofs4face
            for l = 1:ndofs4face
                A4bface[k,l] = dot(basisvals[celldof2facedof[k]][celldof2facedof[k],:],basisvals[celldof2facedof[k]][celldof2facedof[l],:]);
            end
                
            # evaluate Dirichlet data
            boundary_data!(temp,trafo_on_cell(xref[celldof2facedof[k],:]));
            b4bface[k] = dot(temp,basisvals[celldof2facedof[k]][celldof2facedof[k],:]);
        end

        # solve
        val4dofs[dofs4bface] = A4bface\b4bface;
        if norm(A4bface*val4dofs[dofs4bface]-b4bface) > eps(1e3)
            println("WARNING: large residual, boundary data may be inexact");
        end
        end    
    return unique(bdofs)
end

# computes Bestapproximation in approx_norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeBestApproximation!(val4dofs::Array,approx_norm::String ,volume_data!::Function,boundary_data!,FE::AbstractFiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)

    println("\nCOMPUTING BESTAPPROXIMATION")
    println(" |   FE = " * FE.name)
    println(" |ndofs = ", FiniteElements.get_ndofs(FE))
    println(" |");
    println(" |PROGRESS")

    # assemble system 
    @time begin
        print("    |assembling...")
        A, b = assembleSystem(1.0,approx_norm,approx_norm,volume_data!,FE,quadrature_order);
        println("finished")
    end
    
    # apply boundary data
    celldim::Int = size(FE.grid.nodes4cells,2) - 1;
    if (celldim == 1)
        bdofs = computeDirichletBoundaryData!(val4dofs,FE,boundary_data!,false);
    else
        bdofs = computeDirichletBoundaryData!(val4dofs,FE,boundary_data!,true);
    end
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4dofs[bdofs[i]]*dirichlet_penalty;
    end

    # solve
    @time begin
        print("    |solving...")
        try
            val4dofs[:] = A\b;
        catch    
            println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
            try
                val4dofs[:] = Array{typeof(FE.grid.coords4nodes[1]),2}(A)\b;
            catch OverflowError
                println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
                val4dofs[:] = Array{Float64,2}(A)\b;
            end
        end
        println("finished")
    end
    
    # compute residual (exclude bdofs)
    residual = A*val4dofs - b
    residual[bdofs] .= 0
    residual = norm(residual);
    println("    |residual=", residual)
    
    return residual
end

# TODO: has to be rewritten!!!
function computeFEInterpolation!(val4dofs::Array,source_function!::Function,FE::AbstractH1FiniteElement)
    ET = FE.grid.elemtypes[1];
    temp = zeros(Float64,FiniteElements.get_ncomponents(FE));
    xref = FiniteElements.get_xref4dof(FE, ET)
    ndofs4cell = FiniteElements.get_ndofs4elemtype(FE, ET);
    loc2glob_trafo = Grid.local2global(FE.grid,ET)
    dofs = zeros(Int64,ndofs4cell)
    ncomponents = FiniteElements.get_ncomponents(FE);
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    # loop over nodes
    for cell = 1 : size(FE.grid.nodes4cells,1)
        # get trafo
        cell_trafo = loc2glob_trafo(cell)
        
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs,FE,cell,ET)

        # get coefficients
        FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell)

        for k = 1 : ndofs4cell
            x = cell_trafo(xref[k, :]);
            source_function!(temp,x);
            val4dofs[dofs[k],:] = temp;
        end    
    end
end


function eval_FEfunction(coeffs, FE::AbstractH1FiniteElement)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ET = FE.grid.elemtypes[1];
    ndofs4cell = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    dofs = zeros(Int64,ndofs4cell)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    function closure(result, x, xref, cell)
        fill!(result,0.0)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs,FE,cell,ET)

        # get coefficients
        FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell)

        for j = 1 : ndofs4cell
            for k = 1 : ncomponents;
                result[k] += basisvals[j,k] * coeffs[dofs[j]] * coefficients[j,k];
            end    
        end    
    end
end

function eval_L2_interpolation_error!(exact_function!, coeffs_interpolation, FE::AbstractH1FiniteElement)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents);
    dofs = zeros(Int64,ndofs4cell)
    function closure(result, x, xref, cell)
        # evaluate exact function
        exact_function!(result, x);

        # evaluate FE functions
        basisvals = FiniteElements.get_basis_on_elemtype(FE, ET)(xref)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

        # get coefficients
        FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell);

        for j = 1 : ndofs4cell
            for k = 1 : ncomponents;
                result[k] -= basisvals[j,k] * coeffs_interpolation[dofs[j]] * coefficients[j,k];
            end    
        end   
        # square for L2 norm
        for j = 1 : length(result)
            result[j] = result[j]^2
        end    
    end
end


function eval_L2_interpolation_error!(exact_function!, coeffs_interpolation, FE::AbstractHdivFiniteElement)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ET = FE.grid.elemtypes[1]
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents);
    AT = zeros(Float64,2,2)
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, ET)
    det = 0.0;
    dofs = zeros(Int64,ndofs4cell)
    function closure(result, x, xref, cellIndex)
        # evaluate exact function
        exact_function!(result, x);
        
        # subtract nodal interpolation
        det = get_Piola_trafo_on_cell!(AT,cellIndex);
        basisvals[:] = FiniteElements.get_basis_on_elemtype(FE, ET)(xref)

        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs, FE, cellIndex, ET)

        # get coefficients
        FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cellIndex)

        for j = 1 : ndofs4cell
            for k = 1 : ncomponents
                for l = 1: ncomponents
                    result[k] -= AT[k,l] * basisvals[j,l] * coeffs_interpolation[dofs[j]] * coefficients[j,k] / det;
                end    
            end    
        end   
        # square for L2 norm
        for j = 1 : length(result)
            result[j] = result[j]^2
        end    
    end
end


function eval_at_nodes(val4dofs, FE::AbstractH1FiniteElement, offset::Int64 = 0)
    # evaluate at grid points
    ndofs4node = zeros(size(FE.grid.coords4nodes,1))
    ET = FE.grid.elemtypes[1]
    xref4dofs4cell = Grid.get_reference_cordinates(ET)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    nodevals = zeros(size(FE.grid.coords4nodes,1),ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    dofs = zeros(Int64,ndofs4cell)
    for j = 1 : size(xref4dofs4cell,1)
        # get basisvals
        basisvals[:] = FiniteElements.get_basis_on_elemtype(FE,ET)(xref4dofs4cell[j,:])

        for cell = 1 : size(FE.grid.nodes4cells,1)

            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

            # get coefficients
            FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell)
        
            for dof = 1 : ndofs4cell;
                for k = 1 : ncomponents
                    nodevals[FE.grid.nodes4cells[cell,j],k] += basisvals[dof,k]*val4dofs[dofs[dof]]*coefficients[dof,k];
                end   
            end
            ndofs4node[FE.grid.nodes4cells[cell,j]] +=1
        end    
    end
    # average
    for k = 1 : ncomponents
        nodevals[:,k] ./= ndofs4node
    end
    return nodevals
end   

function eval_at_nodes(val4dofs, FE::AbstractHdivFiniteElement, offset::Int64 = 0)
    # evaluate at grid points
    ndofs4node = zeros(size(FE.grid.coords4nodes,1))
    ET = FE.grid.elemtypes[1]
    xref4dofs4cell = Grid.get_reference_cordinates(ET)
    ncomponents = FiniteElements.get_ncomponents(FE);
    ndofs4cell::Int = FiniteElements.get_ndofs4elemtype(FE, ET);
    basisvals = zeros(eltype(FE.grid.coords4nodes),ndofs4cell,ncomponents)
    nodevals = zeros(size(FE.grid.coords4nodes,1),ncomponents)
    coefficients = zeros(Float64,ndofs4cell,ncomponents)
    AT = zeros(Float64,2,2)
    get_Piola_trafo_on_cell! = Grid.local2global_Piola(FE.grid, ET)
    det = 0.0;
    dofs = zeros(Int64, ndofs4cell)
    for j = 1 : size(xref4dofs4cell,1)

        basisvals[:] = FiniteElements.get_basis_on_elemtype(FE, ET)(xref4dofs4cell[j,:])

        for cell = 1 : size(FE.grid.nodes4cells,1)
            # get dofs
            FiniteElements.get_dofs_on_cell!(dofs, FE, cell, ET);

            # get coefficients
            FiniteElements.set_basis_coefficients_on_cell!(coefficients, FE, cell);
            
            # get trafo
            det = get_Piola_trafo_on_cell!(AT,cell);

    
            for dof = 1 : ndofs4cell;
                for k = 1 : ncomponents
                    for l = 1: ncomponents
                        nodevals[FE.grid.nodes4cells[cell,j],k] +=  AT[k,l] * basisvals[dof,l]*val4dofs[dofs[dof]]*coefficients[dof,k] / det;
                    end    
                end   
            end
            ndofs4node[FE.grid.nodes4cells[cell,j]] +=1
        end    
    end
    # average
    for k = 1 : ncomponents
        nodevals[:,k] ./= ndofs4node
    end
    return nodevals
end   

end
