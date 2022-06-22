"""
````
abstract type HDIVRTkENRICH{k,edim} <: AbstractHdivFiniteElement where {edim<:Int}
````

Internal (normal-zero) Hdiv-conforming vector-valued (ncomponents = edim) Raviart-Thomas space of order k
with the additional orthogonality property that their divergences are L2-orthogonal on P_{k-edim+1}.

allowed ElementGeometries:
- Triangle2D
- Tetrahedron3D
"""

## DEFINITION OF RTk BUBBLE SPACES
## contain only that many non divergence-free RTk bubbles that seem to be needed to correct divergence moments for PkxPk-1
## 
abstract type HDIVRTkENRICH{edim,k} <: AbstractHdivFiniteElement where {k<:Int, edim<:Int} end

const _num_RTk_enrich_bubbles = [[2,3,4],[3,9]]
GradientRobustMultiPhysics.get_ncomponents(::Type{<:HDIVRTkENRICH{edim, order}}) where {order, edim} = edim
GradientRobustMultiPhysics.get_ndofs(::Type{ON_CELLS}, FEType::Type{<:HDIVRTkENRICH{edim,order}}, EG::Type{<:AbstractElementGeometry}) where {edim, order} = _num_RTk_enrich_bubbles[edim-1][order]
GradientRobustMultiPhysics.get_polynomialorder(::Type{<:HDIVRTkENRICH{edim, order}}, ::Type{<:AbstractElementGeometry}) where {edim, order} = order + 1
GradientRobustMultiPhysics.get_dofmap_pattern(FEType::Type{<:HDIVRTkENRICH}, ::Type{CellDofs}, EG::Type{<:AbstractElementGeometry}) = "i$(get_ndofs(ON_CELLS, FEType, EG))"

isdefined(FEType::Type{<:HDIVRTkENRICH}, ::Type{<:Triangle2D}) = true
isdefined(FEType::Type{<:HDIVRTkENRICH}, ::Type{<:Tetrahedron3D}) = true

## basis on reference triangle
function GradientRobustMultiPhysics.get_basis(::Type{ON_CELLS}, ::Type{HDIVRTkENRICH{2, order}}, EG::Type{<:Triangle2D}) where {order}
    @assert order in 1:3
    #@show chosen_bubbles
    function closure(refbasis, xref)
        # RT1 bubbles
        # ψ_j is RT0 basis function of j-th edge
        # φ_j is nodal basis function
        # node [3,1,2] is opposite to face [1,2,3]

        # 3
        # |\
        # | \ E2
        # |  \
        # |___\
        # 1 E1 2

        # RT1 cell bubbles
        refbasis[1,1] = xref[2] * xref[1]; refbasis[1,2] = xref[2] * (xref[2]-1)  # = φ_3 ψ_1
        refbasis[2,1] = xref[1] * (xref[1]-1); refbasis[2,2] = xref[1] * xref[2]  # = φ_2 ψ_3

        # minimal selection with L^2 orthogonality: div(*) ⟂ P_{order-1})
        if order == 1
            # just the RT1 bubbles above
        elseif order == 2
            # special bubbles with zero lowest order divergence moments
            for k = 1 : 2
                refbasis[3,k] = (5*(1-xref[1]-xref[2])-2)*(-refbasis[1,k]-refbasis[2,k])
                refbasis[1,k] = (5*xref[2]-2)*refbasis[1,k]
                refbasis[2,k] = (5*xref[1]-2)*refbasis[2,k]
            end
        elseif order == 3
            for k = 1 : 2
                refbasis[3,k] = (7*(1-xref[1]-xref[2])^2 - 6*(1-xref[1]-xref[2]) + 1)*(-refbasis[1,k]-refbasis[2,k])/7
                refbasis[4,k] = -2*xref[1]*xref[2]*refbasis[2,k] + 2//45*(-refbasis[1,k] + 4*refbasis[2,k])
                refbasis[2,k] = (7*xref[1]^2 - 6*xref[1] + 1)*refbasis[2,k]/7
                refbasis[1,k] = (7*xref[2]^2 - 6*xref[2] + 1)*refbasis[1,k]/7
                refbasis[4,k] += (3*refbasis[1,k] + 2*refbasis[2,k] - 3*refbasis[3,k])/70
            end
        end
    end
end 


## basis on reference tetrahedron
function GradientRobustMultiPhysics.get_basis(::Type{ON_CELLS}, ::Type{HDIVRTkENRICH{3, order}}, EG::Type{<:Tetrahedron3D}) where {order}
    @assert order in 1:2
    #@show chosen_bubbles
    function closure(refbasis, xref) 
        # all RT1 bubbles
        refbasis[1,1] = 2*xref[3] * xref[1];      refbasis[1,2] = 2*xref[3] * xref[2];      refbasis[1,3] = 2*xref[3] * (xref[3]-1)
        refbasis[2,1] = 2*xref[2] * xref[1];      refbasis[2,2] = 2*xref[2] * (xref[2]-1);  refbasis[2,3] = 2*xref[2] * xref[3]
        refbasis[3,1] = 2*xref[1] * (xref[1]-1);  refbasis[3,2] = 2*xref[1] * xref[2];      refbasis[3,3] = 2*xref[1] * xref[3]

        if order == 1
            # nothing to add (but enrichment need additional RT0 handled by seperate FESpace/FEVectorBlock)
        elseif order == 2
            for k = 1 : 3
                refbasis[4,k] = (6*(1-xref[1]-xref[2]-xref[3])-1)*refbasis[3,k] + (6*xref[1]-1)*(-refbasis[1,k]-refbasis[2,k]-refbasis[3,k]) # (1,2)
                refbasis[5,k] = (6*(1-xref[1]-xref[2]-xref[3])-1)*refbasis[2,k] + (6*xref[2]-1)*(-refbasis[1,k]-refbasis[2,k]-refbasis[3,k]) # (1,3)
                refbasis[6,k] = (6*(1-xref[1]-xref[2]-xref[3])-1)*refbasis[1,k] + (6*xref[3]-1)*(-refbasis[1,k]-refbasis[2,k]-refbasis[3,k]) # (1,4)
                refbasis[7,k] = (6*xref[1]-1)*refbasis[2,k] + (6*xref[2]-1)*refbasis[3,k] # (2,3)
                refbasis[8,k] = (6*xref[1]-1)*refbasis[1,k] + (6*xref[3]-1)*refbasis[3,k] # (2,4)
                refbasis[9,k] = (6*xref[2]-1)*refbasis[1,k] + (6*xref[3]-1)*refbasis[2,k] # (3,4)
            end
        end
    end
end 
           