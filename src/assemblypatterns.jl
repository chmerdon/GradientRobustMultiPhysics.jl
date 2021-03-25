############################
# AbstractAssemblyPatterns #
############################

abstract type AssemblyPatternType end
abstract type APT_Undefined <: AssemblyPatternType end

# backpack to save all the information needed for assembly
# like element geometries (EG), quadrature formulas (qf), basisevaluators for the FES, ...
# idea is to store this to avoid recomputation in e.g. an in iterative scheme
# also many redundant stuff within assembly of patterns happens here
# like chosing the coressponding basis evaluators, quadrature rules and managing the dofs
struct AssemblyManager{T <: Real}
    xItemDofs::Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}                # DofMaps
    ndofs4EG::Array{Array{Int,1},1}             # ndofs for each finite element on each EG
    nop::Int                                    # number of operators
    qf::Array{QuadratureRule,1}                 # quadrature rules
    basisevaler::Array{FEBasisEvaluator,4}      # finite element basis evaluators
    dii4op::Array{Function,1}
    basisAT::Array{Type{<:AbstractAssemblyType},1}
    citem::Array{Int,1}
    dofitems::Array{Array{Int,1},1}
    EG4dofitem::Array{Array{Int,1},1}          # coordinate 1 in basisevaler
    itempos4dofitem::Array{Array{Int,1},1}      # coordinate 3 in basisevaler
    coeff4dofitem::Array{Array{T,1},1}          
    dofoffset4dofitem::Array{Array{Int,1},1}
    orientation4dofitem::Array{Array{Int,1},1}  # coordinate 4 in basisevaler
end

function update!(AM::AssemblyManager{T}, item::Int) where {T <: Real}
    # get dofitem informations
    if AM.citem[1] != item
        AM.citem[1] = item
        for j = 1 : AM.nop
            AM.dii4op[j](AM.dofitems[j], AM.EG4dofitem[j], AM.itempos4dofitem[j], AM.coeff4dofitem[j], AM.orientation4dofitem[j], AM.dofoffset4dofitem[j], item)
            # update needed basisevaler
            for di = 1 : length(AM.dofitems[j])
                if AM.dofitems[j][di] != 0
                    update!(AM.basisevaler[AM.EG4dofitem[j][di],j,AM.itempos4dofitem[j][di],AM.orientation4dofitem[j][di]], AM.dofitems[j][di])
                end
            end
        end
    end
end

@inline get_basisevaler(AM::AssemblyManager, op::Int, dofitem::Int) = AM.basisevaler[AM.EG4dofitem[op][dofitem],op,AM.itempos4dofitem[op][dofitem],AM.orientation4dofitem[op][dofitem]]
@inline get_qweights(AM::AssemblyManager) = AM.qf[AM.EG4dofitem[1][1]].w
@inline get_ndofs(AM, op::Int, dofitem::Int) = AM.ndofs4EG[op][AM.EG4dofitem[op][dofitem]]
@inline get_maxndofs(AM::AssemblyManager) = maximum.(AM.ndofs4EG)
@inline get_maxndofs(AM::AssemblyManager, op::Int) = maximum(AM.ndofs4EG[op])
@inline function get_maxnqweights(AM::AssemblyManager) 
    maxn::Int = 0
    for j = 1 : length(AM.qf)
        maxn = max(maxn,length(AM.qf[j].w))
    end
    return maxn
end
@inline get_maxdofitems(AM::AssemblyManager) = length.(AM.dofitems)

@inline get_dof(AM::AssemblyManager, op::Int, dofitem::Int, dof_i::Int) = AM.xItemDofs[op][dof_i + AM.dofoffset4dofitem[op][dofitem], AM.dofitems[op][dofitem]]
function get_dofs!(dofs::Array{Int,1}, AM::AssemblyManager, op::Int, dofitem::Int)
    for dof_i = 1 : AM.ndofs4EG[op][AM.EG4dofitem[op][dofitem]]
        dofs[dof_i] = AM.xItemDofs[op][dof_i + AM.dofoffset4dofitem[op][dofitem], AM.dofitems[op][dofitem]]
    end
end
function get_coeffs!(coeffs::Array{T,1}, FE::AbstractArray{T,1}, AM::AssemblyManager, op::Int, dofitem::Int, offset::Int = 0) where {T <: Real}
    for dof_i = 1 : AM.ndofs4EG[op][AM.EG4dofitem[op][dofitem]]
        coeffs[dof_i] = FE[offset + AM.xItemDofs[op][dof_i + AM.dofoffset4dofitem[op][dofitem], AM.dofitems[op][dofitem]]]
    end
end

mutable struct AssemblyPattern{APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType}
    name::String
    FES::Array{FESpace,1}
    operators::Array{DataType,1}
    action::AbstractAction
    regions::Array{Int,1}
    AM::AssemblyManager # hidden stuff needed for assembly
    AssemblyPattern() = new{APT_Undefined, Float64, ON_CELLS}()
    AssemblyPattern{APT, T, AT}() where {APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType} = new{APT,T,AT}()
    AssemblyPattern{APT, T, AT}(name,FES,operators,action,regions) where {APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType}  = new{APT,T,AT}(name, FES,operators,action,regions)
    AssemblyPattern{APT, T, AT}(FES,operators,action,regions) where {APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType}  = new{APT,T,AT}("$APT", FES,operators,action,regions)
end 

# this function decides which basis should be evaluated for the evaluation of an operator
# e.g. Hdiv elements can use the face basis for the evaluation of the normal flux operator,
# but an H1 element must evaluate the cell basis for GradientDisc{Jump} ON_FACES
function DefaultBasisAssemblyType4Operator(operator::Type{<:AbstractFunctionOperator}, patternAT::Type{<:AbstractAssemblyType}, continuity::Type{<:AbstractFiniteElement})
    if patternAT == ON_CELLS
        return ON_CELLS
    elseif patternAT <: Union{<:ON_FACES,<:ON_BFACES}
        if continuity <: AbstractH1FiniteElement
            if QuadratureOrderShift4Operator(operator) == 0
                return patternAT
            else
                return ON_CELLS
            end
        elseif continuity <: AbstractHdivFiniteElement
            if QuadratureOrderShift4Operator(operator) == 0 && operator <: NormalFlux
                return patternAT
            else
                return ON_CELLS
            end
        elseif continuity <: AbstractHcurlFiniteElement
            if QuadratureOrderShift4Operator(operator) == 0 && operator <: TangentFlux
                return patternAT
            else
                return ON_CELLS
            end
        else
            return ON_CELLS
        end
    elseif patternAT <: Union{<:ON_EDGS,<:ON_BEDGES}
        if continuity <: AbstractH1FiniteElement
            if QuadratureOrderShift4Operator(operator) == 0
                return patternAT
            else
                return ON_CELLS
            end
        elseif continuity <: AbstractHdivFiniteElement
            return ON_CELLS
        elseif continuity <: AbstractHcurlFiniteElement
            if QuadratureOrderShift4Operator(operator) == 0 && operator <: TangentFlux
                return patternAT
            else
                return ON_CELLS
            end
        else
            return ON_CELLS
        end
    else return patternAT
    end
end


function DofitemInformation4Operator(FES::FESpace, AT::Type{<:AbstractAssemblyType}, basisAT::Type{<:AbstractAssemblyType}, FO::Type{<:AbstractFunctionOperator})
    # check if operator is discontinuous for this AT
    xgrid = FES.xgrid
    discontinuous = false
    posdt = 0
    for j = 1 : length(FO.parameters)
        if FO.parameters[j] <: DiscontinuityTreatment
            discontinuous = true
            posdt = j
            break;
        end
    end
    if discontinuous
        # call discontinuity handlers
        # for e.g. IdentityDisc, GradientDifsc etc.
        # if AT == ON_FACES: if continuity allows it, assembly ON_FACES two times on each face
        #                    otherwise leads to ON_CELL assembly on neighbouring CELLS
        # if AT == ON_EDGES: todo (should lead to ON_CELL assembly on neighbouring CELLS, morea than two)
        DofitemInformation4Operator(FES, AT, basisAT, FO.parameters[posdt])
    else
        # call standard handlers
        # assembly as specified by AT
        DofitemInformation4Operator(FES, AT, basisAT)
    end
end

# default handlers for continupus operators
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:AbstractAssemblyType}, basisAT::Type{<:AbstractAssemblyType})
    xgrid = FES.xgrid
    xItemGeometries = xgrid[GridComponentGeometries4AssemblyType(AT)]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    # operator is assumed to be continuous, hence only needs to be evaluated on one dofitem = item
    function closure(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, dofoffset4dofitem, item)
        dofitems[1] = item
        itempos4dofitem[1] = 1
        coefficient4dofitem[1] = 1
        # find EG index for geometry
        for j=1:length(EG)
            if xItemGeometries[item] == EG[j]
                EG4dofitem[1] = j
                break;
            end
        end
        return EG4dofitem[1]
    end
    return closure
end

function DofitemInformation4Operator(FES::FESpace, EG, EGdofitems, AT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    return DofitemInformation4Operator(FES, EG, EGdofitems, AT)
end

# special handlers for jump operators with ON_CELL basis
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_FACES}, basisAT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xFaceGeometries = xgrid[FaceGeometries]
    xCellGeometries = xgrid[CellGeometries]
    xCellFaceOrientations = xgrid[CellFaceOrientations]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    EGdofitems = xgrid[GridComponentUniqueGeometries4AssemblyType(basisAT)]
    if DiscType == Jump
        coeff_left = 1
        coeff_right = -1
    elseif DiscType == Average
        coeff_left = 0.5
        coeff_right = 0.5
    end
    # operator is discontinous ON_FACES and needs to be evaluated on the two neighbouring cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, dofoffset4dofitem, item)
        dofitems[1] = xFaceCells[1,item]
        dofitems[2] = xFaceCells[2,item]
        for k = 1 : num_targets(xCellFaces,dofitems[1])
            if xCellFaces[k,dofitems[1]] == item
                itempos4dofitem[1] = k
            end
        end
        orientation4dofitem[1] = xCellFaceOrientations[itempos4dofitem[1], dofitems[1]]
        if dofitems[2] > 0
            for k = 1 : num_targets(xCellFaces,dofitems[2])
                if xCellFaces[k,dofitems[2]] == item
                    itempos4dofitem[2] = k
                end
            end
            orientation4dofitem[2] = xCellFaceOrientations[itempos4dofitem[2], dofitems[2]]
            coefficient4dofitem[1] = coeff_left
            coefficient4dofitem[2] = coeff_right
        else
            coefficient4dofitem[1] = 1
            coefficient4dofitem[2] = 0
            if AT == ON_IFACES
                # if assembly is only on interior faces, ignore boundary faces by setting dofitems to zero
                dofitems[1] = 0
            end
        end

        # find EG index for geometry
        for k = 1 : 2
            if dofitems[k] > 0
                for j=1:length(EGdofitems)
                    if xCellGeometries[dofitems[k]] == EGdofitems[j]
                        EG4dofitem[k] = length(EG) + j
                        break;
                    end
                end
            end
        end
        for j=1:length(EG)
            if xFaceGeometries[item] == EG[j]
                return j
            end
        end
    end
    return closure!
end


# special handlers for jump operators on FACES of broken spaces that otherwise have the necessary continuity (so that they can use the ON_FACES basis)
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_FACES}, basisAT::Type{<:ON_FACES}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xItemGeometries = xgrid[FaceGeometries]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    EGdofitems = xgrid[GridComponentUniqueGeometries4AssemblyType(basisAT)]
    if DiscType == Jump
        coeff_left = 1
        coeff_right = -1
    elseif DiscType == Average
        coeff_left = 0.5
        coeff_right = 0.5
    end
    localndofs = zeros(Int, length(EGdofitems))
    for j = 1 : length(EGdofitems)
        localndofs[j] = get_ndofs(basisAT, typeof(FES).parameters[1], EGdofitems[j])
    end
    # operator is discontinous ON_FACES and needs to be evaluated in the face dofs of the neighbourings cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, dofoffset4dofitem, item)
        if xFaceCells[2,item] > 0
            coefficient4dofitem[1] = coeff_left
            coefficient4dofitem[2] = coeff_right
            dofitems[1] = item
            dofitems[2] = item
        else
            if AT == ON_IFACES
                # if assembly is only on interior faces, ignore boundary faces by setting dofitems to zero
                dofitems[1] = 0
            else
                coefficient4dofitem[1] = 1
                coefficient4dofitem[2] = 0
                dofitems[1] = item
            end
            dofitems[2] = 0
        end
        # find EG index for geometry
        for j=1:length(EG)
            if xItemGeometries[item] == EG[j]
                EG4dofitem[1] = j
                EG4dofitem[2] = j
                dofoffset4dofitem[2] = localndofs[j]
                break;
            end
        end
        return EG4dofitem[1]
    end
    return closure!
end


# special handlers for jump operators with ON_CELL basis
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_BFACES}, basisAT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xBFaceGeometries = xgrid[BFaceGeometries]
    xCellGeometries = xgrid[CellGeometries]
    xBFaces = xgrid[BFaces]
    xCellFaceOrientations = xgrid[CellFaceOrientations]
    EG = xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]
    EGdofitems = xgrid[GridComponentUniqueGeometries4AssemblyType(basisAT)]

    # operator is discontinous ON_FACES and needs to be evaluated on the two neighbouring cells
    function closure!(dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, dofoffset4dofitem, item)
        dofitems[1] = xFaceCells[1,xBFaces[item]]
        dofitems[2] = xFaceCells[2,xBFaces[item]]
        for k = 1 : num_targets(xCellFaces,dofitems[1])
            if xCellFaces[k,dofitems[1]] == xBFaces[item]
                itempos4dofitem[1] = k
            end
        end
        orientation4dofitem[1] = xCellFaceOrientations[itempos4dofitem[1], dofitems[1]]
        coefficient4dofitem[1] = 1

        # find EG index for geometry
        for j=1:length(EG)
            if xCellGeometries[dofitems[1]] == EGdofitems[j]
                EG4dofitem[1] = length(EG) + j
                break;
            end
        end
        for j=1:length(EG)
            if xBFaceGeometries[item] == EG[j]
                return j
            end
        end
    end
    return closure!
end





function prepare_assembly!(AP::AssemblyPattern{APT,T,AT}; FES = "from AP", verbosity::Int = 0) where {APT <: AssemblyPatternType, T<: Real, AT <: AbstractAssemblyType}

    if FES != "from AP"
        FE = FES
    else
        FE = AP.FES
    end

    regions = AP.regions
    bonus_quadorder = AP.action.bonus_quadorder
    operator = AP.operators
    xItemRegions::Union{VectorOfConstants{Int32}, Array{Int32,1}} = FE[1].xgrid[GridComponentRegions4AssemblyType(AT)]
    xItemDofs = Array{Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}},1}(undef,length(FE))
    EG = FE[1].xgrid[GridComponentUniqueGeometries4AssemblyType(AT)]

    # note: EG are the element geometries related to AT (the real integration domains)

    # find out which operators need to evaluate which basis
    # e.g. FaceJump operators that are assembled ON_CELLS
    # and get the corressponding dofmaps
    dofitemAT = Array{Type{<:AbstractAssemblyType},1}(undef,length(FE))
    continuous_operators = []
    discontinuous_operators = []
    broken_operators = []
    discontinuous::Bool = false
    dii4op = Array{Function,1}(undef,length(FE))
    for j=1:length(FE)
        dofitemAT[j] = DefaultBasisAssemblyType4Operator(operator[j], AT, typeof(FE[j]).parameters[1])
        xItemDofs[j] = Dofmap4AssemblyType(FE[j],dofitemAT[j])

        broken_space = FE[j].broken
        # check if operator is requested discontinuous
        discontinuous = false
        for k = 1 : length(operator[j].parameters)
            if typeof(operator[j].parameters[k]) != Int64
                if operator[j].parameters[k] <: DiscontinuityTreatment
                    discontinuous = true
                end
            end
        end


        if dofitemAT[j] != AT
            #println("Operator $(operator[j]) for $(typeof(FE[j])) is evaluated in full discontinuous mode with ON_CELL basis")
            push!(discontinuous_operators,j)
            # this means the assembly uses the full cell dofmaps and assembles two times on the neighbouring cells (e.g. xFaceCells)
        elseif (dofitemAT[j] == AT) && discontinuous && !FE[j].broken
            #println("Operator $(operator[j]) for $(typeof(FE[j])) can be evaluated continuously, but is forced to full discontinuous mode by operator")
            push!(discontinuous_operators,j)
            dofitemAT[j] = ON_CELLS
        elseif (dofitemAT[j] == AT) && FE[j].broken && AT != ON_CELLS && discontinuous
            #println("Operator $(operator[j]) for $(typeof(FE[j])) is evaluated in broken mode using the basis $(dofitemAT[j])")
            push!(broken_operators,j)
            #push!(discontinuous_operators,j)
            #dofitemAT[j] = ON_CELLS
            # this means the assembly uses the broken dofmaps and assembles two times on the same item (with a dof offset)
        else 
            #println("Operator $(operator[j]) for $(typeof(FE[j])) is evaluated in continuous mode using the basis $(dofitemAT[j])")
            push!(continuous_operators,j)
        end
        #println("AT = $AT, dofitemAT[j] = $(dofitemAT[j])")
        dii4op[j] = DofitemInformation4Operator(FE[j], AT, dofitemAT[j], operator[j])
    end    

    # if one of the operators is a face jump operator we also need the element geometries 
    # of the neighbouring cells
    EGoffset = length(EG)
    EGdofitem = []
    if length(discontinuous_operators) > 0
        EGdofitem = FE[1].xgrid[GridComponentUniqueGeometries4AssemblyType(dofitemAT[discontinuous_operators[1]])]
    end

    # find proper quadrature QuadratureRules
    # and construct matching FEBasisEvaluators
    # dimension 1 = id of element geometry (combination of integration domains/dofitem domains)
    # dimension 2 = id finite element
    # dimension 3 = position if integration domain in superset dofitem (if jumping operator)
    # dimension 4 = left or right side (if jumping operator)
    qf = Array{QuadratureRule,1}(undef,length(EG) + length(EGdofitem))
    quadorder = 0
    ndofs4EG = Array{Array{Int,1},1}(undef,length(FE))
    for e = 1 : length(FE)
        ndofs4EG[e] = zeros(Int,length(EG)+length(EGdofitem))
    end

    ## get basis evaluators
    maxfaces = 0
    maxorientations = 0
    for j = 1 : length(EGdofitem)        
        maxfaces = max(maxfaces,nfaces_for_geometry(EGdofitem[j]))
    end
    if length(discontinuous_operators) > 0
        for j = 1 : length(EG)  
            maxorientations = max(maxorientations, length(xrefFACE2xrefOFACE(EG[j])))
        end
    end
    basisevaler = Array{FEBasisEvaluator,4}(undef, length(EG) + length(EGdofitem), length(FE), (length(discontinuous_operators) > 0) ? maxfaces : 1, (length(discontinuous_operators) > 0) ? maxorientations : 1)

    ## first position: basis evaluator of operators on assembly geometry (but only used in continuous or broken mode)
    for j = 1 : length(EG)
        quadorder = bonus_quadorder
        for k = 1 : length(FE)
            quadorder += get_polynomialorder(eltype(FE[k]), EG[j]) + QuadratureOrderShift4Operator(operator[k])
        end
        quadorder = max(quadorder,0)          
        qf[j] = QuadratureRule{T,EG[j]}(quadorder);
        # choose quadrature order for all finite elements
        for k = 1 : length(FE)
          #  if dofitemAT[k] == AT
                if k > 1 && FE[k] == FE[1] && operator[k] == operator[1]
                    basisevaler[j,k,1,1] = basisevaler[j,1,1,1] # e.g. for symmetric bilinearforms
                elseif k > 2 && FE[k] == FE[2] && operator[k] == operator[2]
                    basisevaler[j,k,1,1] = basisevaler[j,2,1,1]
                else    
                    basisevaler[j,k,1,1] = FEBasisEvaluator{T,eltype(FE[k]),EG[j],operator[k],AT}(FE[k], qf[j]; verbosity = verbosity - 1)
                end    
                ndofs4EG[k][j] = size(basisevaler[j,k,1,1].cvals,2)
         #   end
        end
    end        

    # assign additional basisevaluators for discontinuous operators that need to evaluate a larger basis e.g. ON_CELLS
    # of each unique cell geometry (in EGdofitem) quadrature points of their face geometry are mapped to
    # quadrature points on the cell (NOTE: We assume that all faces of an EGdofitem are of the same shape)
    if length(discontinuous_operators) > 0
        for j = 1 : length(EGdofitem)
            quadorder = bonus_quadorder
            for k = 1 : length(FE)
                quadorder += get_polynomialorder(eltype(FE[k]), EGdofitem[j]) + QuadratureOrderShift4Operator(operator[k])
            end
            quadorder = max(quadorder,0)        
            nfaces4cell = nfaces_for_geometry(EGdofitem[j])
            EGface = facetype_of_cellface(EGdofitem[j], 1)
            EGfaceid = 0
            for f = 1 : length(EG)
                if EG[f] == EGface
                    EGfaceid = f
                    break;
                end
            end
            # load quadrature rule for face
            qf4face = qf[EGfaceid]
            xrefdim = dim_element(EGface) + 1

            # generate new quadrature rules on neighbouring cells
            # where quadrature points of face are mapped to quadrature points of cells
            qf[EGoffset + j] = SQuadratureRule{T,EGdofitem[j],xrefdim,length(qf4face.xref)}(qf4face.name * " (shape faces)",Array{Array{T,1},1}(undef,length(qf4face.xref)),qf4face.w)
            for k in discontinuous_operators
                xrefFACE2CELL = xrefFACE2xrefCELL(EGdofitem[j])
                EGface = facetype_of_cellface(EGdofitem[j], 1)
                xrefFACE2OFACE = xrefFACE2xrefOFACE(EGface)
                for f = 1 : nfaces4cell, orientation = 1 : length(xrefFACE2OFACE)
                    for i = 1 : length(qf4face.xref)
                        qf[EGoffset + j].xref[i] = SVector{xrefdim,T}(xrefFACE2CELL[f](xrefFACE2OFACE[orientation](qf4face.xref[i])))
                        #println("face $f orientation $orientation : mapping  $(qf4face.xref[i]) to $(qf[EGoffset + j].xref[i])")
                    end
                    basisevaler[EGoffset + j,k,f,orientation] = FEBasisEvaluator{T,eltype(FE[k]),EGdofitem[j],operator[k],dofitemAT[k]}(FE[k], qf[EGoffset + j]; verbosity = verbosity - 1)
                end
                ndofs4EG[k][EGoffset+j] = size(basisevaler[EGoffset + j,k,1,1].cvals,2)
            end
        end

        # append EGdofitem to EG
        EG = [EG, EGdofitem]
    end

    if verbosity > 0
        println("  Preparing assembly for $APT")
        println("     regions = $(AP.regions)")
        println("      action = $(typeof(AP.action))")
        println("          EG = $EG")
        println("\n  List of arguments FEType / operator / ndofs4EG:")
        for k = 1 : length(FE)
            println("      ($k) $(FE[k].name) / $(operator[k]) / $(ndofs4EG[k])")
        end    
        if verbosity > 1
            for j = 1 : length(EG)
                println("\nQuadratureRule [$j] for $(EG[j]):")
                Base.show(qf[j])
            end
        end
    end

    # generate assembly manager
    dofitems = Array{Array{Int,1},1}(undef, length(FE))
    EG4dofitem = Array{Array{Int,1},1}(undef, length(FE))        
    itempos4dofitem = Array{Array{Int,1},1}(undef, length(FE))    
    coeff4dofitem = Array{Array{T,1},1}(undef, length(FE))          
    dofoffset4dofitem = Array{Array{Int,1},1}(undef, length(FE))
    orientation4dofitem = Array{Array{Int,1},1}(undef, length(FE))  
    for j = 1 : length(FE)
        dofitems[j] = (j in continuous_operators) ? zeros(Int,1) : zeros(Int,2)
        EG4dofitem[j] = [1,1]
        itempos4dofitem[j] = [1,1]
        coeff4dofitem[j] = ones(T,2)
        dofoffset4dofitem[j] = zeros(Int,2)
        orientation4dofitem[j] = [1,1]
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], dofitemAT[j])
    end

    AP.AM = AssemblyManager{T}(xItemDofs,ndofs4EG,length(FE),qf,basisevaler,dii4op,dofitemAT,[0],dofitems,EG4dofitem,itempos4dofitem,coeff4dofitem,dofoffset4dofitem,orientation4dofitem)
end

# each assembly pattern is in its own file
include("assemblypatterns/itemintegrator.jl")
include("assemblypatterns/linearform.jl")
include("assemblypatterns/bilinearform.jl")
include("assemblypatterns/trilinearform.jl")
include("assemblypatterns/multilinearform.jl")
include("assemblypatterns/nonlinearform.jl")