"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = edim) Raviart-Thomas space of order 1

allowed ElementGeometries:
- Triangle2D
"""
abstract type HDIVRT1{edim} <: AbstractHdivFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HDIVRT1}) = FEType.parameters[1]

get_polynomialorder(::Type{<:HDIVRT1{2}}, ::Type{<:AbstractElementGeometry1D}) = 1;
get_polynomialorder(::Type{<:HDIVRT1{2}}, ::Type{<:AbstractElementGeometry2D}) = 2;


function init!(FES::FESpace{FEType}; dofmap_needed = true) where {FEType <: HDIVRT1}
    ncomponents = get_ncomponents(FEType)
    FES.name = "RT1 (H1, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    ncells = num_sources(FES.xgrid[CellNodes])
    FES.ndofs = 2*(nfaces + ncells)

    # register coefficients
    FES.xCellFaceSigns = FES.xgrid[CellFaceSigns]
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeCELL}) where {FEType <: HDIVRT1}
    xCellFaces = FES.xgrid[CellFaces]
    xCellGeometries = FES.xgrid[CellGeometries]
    dofs4item = zeros(Int32,2+2*max_num_targets_per_source(xCellFaces))
    ncells = num_sources(xCellFaces)
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xCellDofs = VariableTargetAdjacency(Int32)
    nfaces4item = 0
    for cell = 1 : ncells
        nfaces4item = num_targets(xCellFaces,cell)
        # face functions
        for k = 1 : nfaces4item
            dofs4item[2*k-1] = xCellFaces[k,cell]
            dofs4item[2*k] = xCellFaces[k,cell] + nfaces
        end
        # interior cell functions
        dofs4item[2*nfaces4item+1] = 2*nfaces + 2*cell-1
        dofs4item[2*nfaces4item+2] = 2*nfaces + 2*cell
        append!(xCellDofs,dofs4item[1:(2+2*nfaces4item)])
    end
    # save dofmap
    FES.CellDofs = xCellDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeFACE}) where {FEType <: HDIVRT1}
    nfaces = num_sources(FES.xgrid[FaceNodes])
    xFaceDofs = VariableTargetAdjacency(Int32)
    for face = 1 : nfaces
        append!(xFaceDofs,face.+ [0 nfaces])
    end
    # save dofmap
    FES.FaceDofs = xFaceDofs
end

function init_dofmap!(FES::FESpace{FEType}, ::Type{AssemblyTypeBFACE}) where {FEType <: HDIVRT1}
    xBFaces = FES.xgrid[BFaces]
    nfaces = num_sources(FES.xgrid[FaceNodes])
    nbfaces = length(xBFaces)
    xBFaceDofs = VariableTargetAdjacency(Int32)
    for bface = 1: nbfaces
        append!(xBFaceDofs,xBFaces[bface] .+ [0 nfaces])
    end
    # save dofmap
    FES.BFaceDofs = xBFaceDofs
end


function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{<:HDIVRT1}, exact_function!::Function; dofs = [], bonus_quadorder::Int = 0)
     # integrate normal flux of exact_function over edges
    ncomponents = get_ncomponents(eltype(FE))
    xFaceNormals = FE.xgrid[FaceNormals]
    nfaces = num_sources(xFaceNormals)
    function normalflux_eval()
        temp = zeros(Float64,ncomponents)
        function closure(result, x, face, xref)
            exact_function!(temp,x)
            result[1] = 0.0
            for j = 1 : ncomponents
                result[1] += temp[j] * xFaceNormals[j,face]
            end 
        end   
    end   
    integrate!(Target, FE.xgrid, AssemblyTypeFACE, normalflux_eval(), bonus_quadorder, 1; item_dependent_integrand = true)
    
    # integrate first moment of normal flux of exact_function over edges
    function normalflux2_eval()
        temp = zeros(Float64,ncomponents)
        function closure(result, x, face, xref)
            exact_function!(temp,x)
            result[1] = 0.0
            for j = 1 : ncomponents
                result[1] += temp[j] * xFaceNormals[j,face]
            end
            result[1] *= -(xref[1] - 1//2)
        end   
    end   
    integrate!(Target, FE.xgrid, AssemblyTypeFACE, normalflux2_eval(), bonus_quadorder + 1, 1; item_dependent_integrand = true, index_offset = nfaces)

    # set values of interior RT1 functions by integrating over cell
    # they are chosen such that integral mean of exact function is preserved on each cell
    ncells = num_sources(FE.xgrid[CellNodes])
    xCellVolumes = FE.xgrid[CellVolumes]
    xCellDofs = FE.CellDofs
    means = zeros(Float64,2,ncells)
    integrate!(means, FE.xgrid, AssemblyTypeCELL, exact_function!, bonus_quadorder, 2)
    qf = QuadratureRule{Float64,Triangle2D}(2)
    FEB = FEBasisEvaluator{Float64,eltype(FE),Triangle2D,Identity,AssemblyTypeCELL}(FE, qf)

    basisval = zeros(Float64,2)
    IMM = zeros(Float64,2,2)
    interiordofs = [xCellDofs[7,1],xCellDofs[8,1]]
    for cell = 1 : ncells
        update!(FEB,cell)
        # compute mean value of facial RT1 dofs
        for dof = 1 : 6
            for i = 1 : length(qf.w)
                eval!(basisval,FEB, dof, i)
                for k = 1 : 2
                    means[k,cell] -= basisval[k] * Target[xCellDofs[dof,cell]] * xCellVolumes[cell] * qf.w[i]
                end
            end
        end
        # compute mss matrix of interior dofs
        fill!(IMM,0)
        for dof = 1:2
            for i = 1 : length(qf.w)
                eval!(basisval,FEB, 6 + dof, i)
                for k = 1 : 2
                    IMM[k,dof] += basisval[k] * xCellVolumes[cell] * qf.w[i]
                end
            end
            interiordofs[dof] = xCellDofs[6 + dof,cell] 
        end
        Target[interiordofs] = IMM\means[:,cell]
    end


end


function get_basis_normalflux_on_face(::Type{<:HDIVRT1}, ::Type{<:AbstractElementGeometry})
    function closure(xref)
        return [1.0 # normal-flux of RT0 function on single face
                -12*(xref[1]-0.5)]; # linear normal-flux of RT1 function
    end
end

function get_basis_on_cell(::Type{HDIVRT1{2}}, ::Type{<:Triangle2D})
    function closure(xref)
        temp = 0.5 - xref[1] - xref[2]
        a = [xref[1] xref[2]-1.0];
        b = [xref[1] xref[2]];
        c = [xref[1]-1.0 xref[2]];
        return [a;
                (12*temp).*a;
                b;
                (12*(xref[1] - 0.5)).*b;
                c;
                (12*(xref[2] - 0.5)).*c;
                12*xref[2].*a
                12*xref[1].*c]
    end
end


function get_coefficients_on_cell!(coefficients, FE::FESpace{<:HDIVRT1}, EG::Type{<:AbstractElementGeometry}, cell::Int)
    # multiplication with normal vector signs
    fill!(coefficients,1)
    for j = 1 : nfaces_for_geometry(EG),  k = 1 : size(coefficients,1)
        coefficients[k,2*j-1] = FE.xCellFaceSigns[j,cell];
    end
end    
