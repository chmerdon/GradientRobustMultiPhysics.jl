struct CELL_STOKES <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_STOKES}, FE_velocity::FiniteElements.AbstractFiniteElement, FE_pressure::FiniteElements.AbstractFiniteElement, nu::Real = 1.0, pressure_diagonal = 1e-14)
    
    # get quadrature formula
    T = eltype(FE_velocity.grid.coords4nodes);
    ET = FE_velocity.grid.elemtypes[1]
    quadorder = maximum([FiniteElements.get_polynomial_order(FE_pressure) + FiniteElements.get_polynomial_order(FE_velocity)-1, 2*(FiniteElements.get_polynomial_order(FE_velocity)-1)]);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cell_velocity::Int = FiniteElements.get_ndofs4elemtype(FE_velocity, ET);
    ndofs4cell_pressure::Int = FiniteElements.get_ndofs4elemtype(FE_pressure, ET);
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    ncomponents::Int = FiniteElements.get_ncomponents(FE_velocity);
    FEbasis_velocity = FiniteElements.FEbasis_caller(FE_velocity, qf, true);
    FEbasis_pressure = FiniteElements.FEbasis_caller(FE_pressure, qf, false);
    basisvals_pressure = zeros(Float64,ndofs4cell_pressure,1)
    gradients_velocity = zeros(Float64,ndofs4cell_velocity,ncomponents*ncomponents);
    dofs_pressure = zeros(Int64,ndofs4cell_pressure)
    dofs_velocity = zeros(Int64,ndofs4cell_velocity)
    diagonal_entries = [1, 4]

    # Assembly rest of Stokes operators
    # Du*Dv + div(u)*q + div(v)*p
    for cell = 1 : size(FE_velocity.grid.nodes4cells,1);
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis_pressure, cell)
        FiniteElements.updateFEbasis!(FEbasis_velocity, cell)
      
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs_velocity,FE_velocity, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofs_pressure,FE_pressure, cell, ET);
        dofs_pressure .+= ndofs_velocity;
      
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals_pressure, FEbasis_pressure, i)
            FiniteElements.getFEbasisgradients4qp!(gradients_velocity, FEbasis_velocity, i)

            for dof_i = 1 : ndofs4cell_velocity
            
                # stiffness matrix for velocity
                for dof_j = 1 : ndofs4cell_velocity
                    temp = 0.0;
                    for k = 1 : size(gradients_velocity,2)
                        temp += gradients_velocity[dof_i,k] * gradients_velocity[dof_j,k];
                    end
                    temp *= nu * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    A[dofs_velocity[dof_i],dofs_velocity[dof_j]] += temp;    
                end
                # pressure x (-div) velocity
                for dof_j = 1 : ndofs4cell_pressure
                    temp = 0.0;
                    for c = 1 : length(diagonal_entries)
                        temp -= gradients_velocity[dof_i,diagonal_entries[c]];
                    end    
                    temp *= basisvals_pressure[dof_j] * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    A[dofs_velocity[dof_i],dofs_pressure[dof_j]] += temp;
                    A[dofs_pressure[dof_j],dofs_velocity[dof_i]] += temp;
                end
            end    

            # pressure x pressure
            if (pressure_diagonal > 0)
                for dof_i = 1 : ndofs4cell_pressure
                    A[dofs_pressure[dof_i],dofs_pressure[dof_i]] += pressure_diagonal * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                end
            end
        end  
    end
    #end  
end


function assemble_operator!(A::ExtendableSparseMatrix, B::ExtendableSparseMatrix,::Type{CELL_STOKES}, FE_velocity::FiniteElements.AbstractFiniteElement, FE_pressure::FiniteElements.AbstractFiniteElement, nu::Real = 1.0, symmetric_gradient = false)
    
    # get quadrature formula
    T = eltype(FE_velocity.grid.coords4nodes);
    ET = FE_velocity.grid.elemtypes[1]
    quadorder = maximum([FiniteElements.get_polynomial_order(FE_pressure) + FiniteElements.get_polynomial_order(FE_velocity)-1, 2*(FiniteElements.get_polynomial_order(FE_velocity)-1)]);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs4cell_velocity::Int = FiniteElements.get_ndofs4elemtype(FE_velocity, ET);
    ndofs4cell_pressure::Int = FiniteElements.get_ndofs4elemtype(FE_pressure, ET);
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    ndofs_pressure = FiniteElements.get_ndofs(FE_pressure);
    ndofs = ndofs_velocity + ndofs_pressure;
    ncomponents::Int = FiniteElements.get_ncomponents(FE_velocity);
    FEbasis_velocity = FiniteElements.FEbasis_caller(FE_velocity, qf, true);
    FEbasis_pressure = FiniteElements.FEbasis_caller(FE_pressure, qf, false);
    basisvals_pressure = zeros(Float64,ndofs4cell_pressure,ncomponents)
    gradients_velocity = zeros(Float64,ndofs4cell_velocity,ncomponents*ncomponents);
    dofs_pressure = zeros(Int64,ndofs4cell_pressure)
    dofs_velocity = zeros(Int64,ndofs4cell_velocity)
    diagonal_entries = [1, 4]
    transposed_entries = [1, 3, 2, 4]
    # Assembly rest of Stokes operators
    # Du*Dv + div(u)*q + div(v)*p
    for cell = 1 : size(FE_velocity.grid.nodes4cells,1);
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis_pressure, cell)
        FiniteElements.updateFEbasis!(FEbasis_velocity, cell)
      
        # get dofs
        FiniteElements.get_dofs_on_cell!(dofs_velocity,FE_velocity, cell, ET);
        FiniteElements.get_dofs_on_cell!(dofs_pressure,FE_pressure, cell, ET);
      
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals_pressure, FEbasis_pressure, i)
            FiniteElements.getFEbasisgradients4qp!(gradients_velocity, FEbasis_velocity, i)

            for dof_i = 1 : ndofs4cell_velocity
            
                # stiffness matrix for velocity
                for dof_j = 1 : ndofs4cell_velocity
                    temp = 0.0;
                    for k = 1 : size(gradients_velocity,2)
                        temp += gradients_velocity[dof_i,k] * gradients_velocity[dof_j,k];
                    end
                    if symmetric_gradient
                        for k = 1 : size(gradients_velocity,2)
                            temp += gradients_velocity[dof_i,transposed_entries[k]] * gradients_velocity[dof_j,k];
                        end
                        temp *= 1//2 * nu * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    else    
                        temp *= nu * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    end    
                    A[dofs_velocity[dof_i],dofs_velocity[dof_j]] += temp;
                end
                # pressure x (-div) velocity
                for dof_j = 1 : ndofs4cell_pressure
                    temp = 0.0;
                    for c = 1 : length(diagonal_entries)
                        temp -= gradients_velocity[dof_i,diagonal_entries[c]];
                    end    
                    temp *= basisvals_pressure[dof_j] * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    B[dofs_velocity[dof_i],dofs_pressure[dof_j]] += temp;
                end
            end    
        end  
    end
    #end  
end