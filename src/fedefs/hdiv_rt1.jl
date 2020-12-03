"""
$(TYPEDEF)

Hdiv-conforming vector-valued (ncomponents = edim) Raviart-Thomas space of order 1

allowed ElementGeometries:
- Triangle2D
"""
abstract type HDIVRT1{edim} <: AbstractHdivFiniteElement where {edim<:Int} end

get_ncomponents(FEType::Type{<:HDIVRT1}) = FEType.parameters[1]
get_ndofs_on_face(FEType::Type{<:HDIVRT1}, EG::Type{<:AbstractElementGeometry1D}) = 2
get_ndofs_on_cell(FEType::Type{<:HDIVRT1}, EG::Type{<:Triangle2D}) = 2*nfaces_for_geometry(EG) + 2

get_polynomialorder(::Type{<:HDIVRT1{2}}, ::Type{<:AbstractElementGeometry1D}) = 1;
get_polynomialorder(::Type{<:HDIVRT1{2}}, ::Type{<:AbstractElementGeometry2D}) = 2;

get_dofmap_pattern(FEType::Type{<:HDIVRT1{2}}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "f2i2"
get_dofmap_pattern(FEType::Type{<:HDIVRT1{2}}, ::Type{FaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i2"
get_dofmap_pattern(FEType::Type{<:HDIVRT1{2}}, ::Type{BFaceDofs}, EG::Type{<:AbstractElementGeometry}) = "i2"

function init!(FES::FESpace{FEType}) where {FEType <: HDIVRT1}
    ncomponents = get_ncomponents(FEType)
    FES.name = "RT1 (Hdiv, $(ncomponents)d)"

    # count number of dofs
    nfaces = num_sources(FES.xgrid[FaceNodes])
    ncells = num_sources(FES.xgrid[CellNodes])
    FES.ndofs = 2*(nfaces + ncells)
end



function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_FACES}, exact_function!; items = [], time = 0) where {FEType <: HDIVRT1}
    ncomponents = get_ncomponents(FEType)
    if items == []
        items = 1 : num_sources(FE.xgrid[FaceNodes])
    end

   # integrate normal flux of exact_function over edges
   xFaceNormals = FE.xgrid[FaceNormals]
   nfaces = num_sources(xFaceNormals)
   function normalflux_eval()
       temp = zeros(Float64,ncomponents)
       function closure(result, x, face)
            eval!(temp, exact_function!, x, time)
            result[1] = 0.0
            for j = 1 : ncomponents
               result[1] += temp[j] * xFaceNormals[j,face]
            end 
       end   
   end   
   edata_function = ExtendedDataFunction(normalflux_eval(), [1, ncomponents]; dependencies = "XI", quadorder = exact_function!.quadorder)
   integrate!(Target, FE.xgrid, ON_FACES, edata_function; items = items)
   
   # integrate first moment of normal flux of exact_function over edges
   function normalflux2_eval()
       temp = zeros(Float64,ncomponents)
       function closure(result, x, face, xref)
            eval!(temp, exact_function!, x, time)
            result[1] = 0.0
            for j = 1 : ncomponents
               result[1] += temp[j] * xFaceNormals[j,face]
            end
            result[1] *= (xref[1] - 1//2)
       end   
   end   
   edata_function2 = ExtendedDataFunction(normalflux2_eval(), [1, ncomponents]; dependencies = "XIL", quadorder = exact_function!.quadorder + 1)
   integrate!(Target, FE.xgrid, ON_FACES, edata_function2; items = items, index_offset = nfaces)

end

function interpolate!(Target::AbstractArray{<:Real,1}, FE::FESpace{FEType}, ::Type{ON_CELLS}, exact_function!; items = [], time = 0) where {FEType <: HDIVRT1}
    # delegate cell faces to face interpolation
    subitems = slice(FE.xgrid[CellFaces], items)
    interpolate!(Target, FE, ON_FACES, exact_function!; items = subitems)

    # set values of interior RT1 functions by integrating over cell
    # they are chosen such that integral mean of exact function is preserved on each cell
    ncells = num_sources(FE.xgrid[CellNodes])
    xCellVolumes = FE.xgrid[CellVolumes]
    xCellDofs = FE.dofmaps[CellDofs]
    means = zeros(Float64,2,ncells)
    integrate!(means, FE.xgrid, ON_CELLS, exact_function!)
    qf = QuadratureRule{Float64,Triangle2D}(2)
    FEB = FEBasisEvaluator{Float64,eltype(FE),Triangle2D,Identity,ON_CELLS}(FE, qf)
    if items == []
        items = 1 : ncells
    end

    basisval = zeros(Float64,2)
    IMM = zeros(Float64,2,2)
    interiordofs = [xCellDofs[7,1],xCellDofs[8,1]]
    for cell in items
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
    function closure(refbasis,xref)
        refbasis[1,1] = 1                # normal-flux of RT0 function on single face
        refbasis[2,1] = 12*(xref[1]-1//2) # linear normal-flux of RT1 function
    end
end

function get_basis_on_cell(::Type{HDIVRT1{2}}, ::Type{<:Triangle2D})
    function closure(refbasis,xref)
        temp = 1//2 - xref[1] - xref[2]
        # RT0 basis
        refbasis[1,:] .= [xref[1], xref[2]-1];
        refbasis[3,:] .= [xref[1], xref[2]];
        refbasis[5,:] .= [xref[1]-1, xref[2]];
        # additional face basis functions
        refbasis[2,:] .= -12*temp .* refbasis[1,:];
        refbasis[4,:] .= -(12*(xref[1] - 1//2)) .* refbasis[3,:];
        refbasis[6,:] .= -(12*(xref[2] - 1//2)) .* refbasis[5,:];
        # interior functions
        refbasis[7,:] .= 12*xref[2] .* refbasis[1,:];
        refbasis[8,:] .= 12*xref[1] .* refbasis[5,:];
    end
end


function get_coefficients_on_cell!(FE::FESpace{<:HDIVRT1}, EG::Type{<:AbstractElementGeometry})
    xCellFaceSigns = FE.xgrid[CellFaceSigns]
    nfaces = nfaces_for_geometry(EG)
    function closure(coefficients, cell)
        fill!(coefficients,1.0)
        # multiplication with normal vector signs (only RT0)
        for j = 1 : nfaces,  k = 1 : size(coefficients,1)
            coefficients[k,2*j-1] = xCellFaceSigns[j,cell];
        end
        return nothing
    end
end    
