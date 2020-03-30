struct CELL_DIVFREE_UdotV <: FiniteElements.AbstractFEOperator end

function assemble_operator!(A::ExtendableSparseMatrix,::Type{CELL_DIVFREE_UdotV}, FE_velocity::FiniteElements.AbstractFiniteElement, FE_pressure::FiniteElements.AbstractFiniteElement, factor = 1.0, pressure_diagonal = 1e-14)
    
    # get quadrature formula
    T = eltype(FE_velocity.grid.coords4nodes);
    ET = FE_velocity.grid.elemtypes[1]
    quadorder = maximum([FiniteElements.get_polynomial_order(FE_pressure) + FiniteElements.get_polynomial_order(FE_velocity)-1, 2*(FiniteElements.get_polynomial_order(FE_velocity))]);
    qf = QuadratureFormula{T,typeof(ET)}(quadorder);
    
    # generate caller for FE basis functions
    ndofs_velocity = FiniteElements.get_ndofs(FE_velocity);
    FEbasis_velocity = FiniteElements.FEbasis_caller(FE_velocity, qf, true);
    FEbasis_pressure = FiniteElements.FEbasis_caller(FE_pressure, qf, false);
    basisvals_pressure = zeros(Float64,FEbasis_pressure.ndofs4item,1)
    basisvals_velocity = zeros(Float64,FEbasis_velocity.ndofs4item,FEbasis_velocity.ncomponents)
    divergences_velocity = zeros(Float64,FEbasis_velocity.ndofs4item,1);
    diagonal_entries = [1, 4]

    # Assembly rest of Stokes operators
    # u*v + div(u)*q + div(v)*p
    temp = 0.0
    for cell = 1 : size(FE_velocity.grid.nodes4cells,1);
        # update FEbasis on cell
        FiniteElements.updateFEbasis!(FEbasis_pressure, cell)
        FiniteElements.updateFEbasis!(FEbasis_velocity, cell)

        # add offset to pressure dofs
        FEbasis_pressure.current_dofs .+= ndofs_velocity;
      
        for i in eachindex(qf.w)
            # get FE basis at quadrature point
            FiniteElements.getFEbasis4qp!(basisvals_pressure, FEbasis_pressure, i)
            FiniteElements.getFEbasis4qp!(basisvals_velocity, FEbasis_velocity, i)
            FiniteElements.getFEbasisdivergence4qp!(divergences_velocity, FEbasis_velocity, i)

            for dof_i = 1 : FEbasis_velocity.ndofs4item
            
                # stiffness matrix for velocity
                for dof_j = 1 : FEbasis_velocity.ndofs4item
                    temp = 0.0;
                    for k = 1 : FEbasis_velocity.ncomponents
                        temp += basisvals_velocity[dof_i,k] * basisvals_velocity[dof_j,k];
                    end
                    temp *= factor * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    A[FEbasis_velocity.current_dofs[dof_i],FEbasis_velocity.current_dofs[dof_j]] += temp;
                end
                # pressure x (-div) velocity
                for dof_j = 1 : FEbasis_pressure.ndofs4item
                    temp = -divergences_velocity[dof_i] * basisvals_pressure[dof_j] * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                    A[FEbasis_velocity.current_dofs[dof_i],FEbasis_pressure.current_dofs[dof_j]] += temp;
                    A[FEbasis_pressure.current_dofs[dof_j],FEbasis_velocity.current_dofs[dof_i]] += temp;
                end
            end    

            # pressure x pressure
            if (pressure_diagonal > 0)
                for dof_i = 1 : FEbasis_pressure.ndofs4item
                    A[FEbasis_pressure.current_dofs[dof_i],FEbasis_pressure.current_dofs[dof_i]] += pressure_diagonal * qf.w[i] * FE_velocity.grid.volume4cells[cell];
                end
            end
        end  
    end
    #end  
end