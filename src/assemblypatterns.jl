############################
# AbstractAssemblyPatterns #
############################


"""
$(TYPEDEF)

root type for assembly pattern types
"""
abstract type AssemblyPatternType end

abstract type APT_Undefined <: AssemblyPatternType end

abstract type DIIType end
abstract type DIIType_continuous <: DIIType end
abstract type DIIType_discontinuous{DiscType,AT,basisAT} <: DIIType end
abstract type DIIType_broken{DiscType,AT,basisAT} <: DIIType end

# backpack to save all the information needed for assembly
# like element geometries (EG), quadrature formulas (qf), basisevaluators for the FES, ...
# idea is to store this to avoid recomputation in e.g. an in iterative scheme
# also many redundant stuff within assembly of patterns happens here
# like chosing the coressponding basis evaluators, quadrature rules and managing the dofs
struct AssemblyManager{T <: Real}
    xItemDofs::Array{DofMapTypes{Int32},1}                # DofMaps
    ndofs4EG::Array{Array{Int,1},1}             # ndofs for each finite element on each EG
    nop::Int                                    # number of operators
    qf::Array{QuadratureRule,1}                 # quadrature rules
    basisevaler::Array{AbstractFEBasisEvaluator{T},4}      # finite element basis evaluators
    dii4op_types::Array{DataType,1}
    basisAT::Array{Type{<:AbstractAssemblyType},1}                      
    citem::Base.RefValue{Int}                                           # current item
    dofitems::Array{Array{Int,1},1}                                     # dofitems needed to visit to evaluate operator
    EG4dofitem::Array{Array{Int,1},1}                                   # id to [EG,dEG] = coordinate 1 in basisevaler
    itempos4dofitem::Array{Array{Int,1},1}                              # local EG position in dEG = coordinate 3 in basisevaler
    coeff4dofitem::Array{Array{T,1},1}                                  # coefficients for operators (needed for jump/average operators)
    dofoffset4dofitem::Array{Array{Int,1},1}                            # offesets (needed for broken FESpaces)
    orientation4dofitem::Array{Array{Int,1},1}                          # local orientation of EG in dEG = coordinate 4 in basisevaler
    EG::Array{Type{<:AbstractElementGeometry},1}                        # unique item geometries that may appear 
    dEG::Array{Type{<:AbstractElementGeometry},1}                       # unique dofitem geometries that may appear (for each operator)
    xItemGeometries::GridEGTypes                                        # item geometries over which is assembled
    xDofItemGeometries::Array{GridEGTypes,1}                       # dofitem geometries over which basis is evaluated (for each operator)
    xDofItems4Item::Array{Union{Nothing,GridAdjacencyTypes{Int32}},1}       # dofitems <> items adjacency relationship (for each operator)
    xItemInDofItems::Array{Union{Nothing,GridAdjacencyTypes{Int32}},1}      # where is the item locally in dofitems
    xDofItemItemOrientations::Array{Union{Nothing,GridAdjacencyTypes{Int32}},1} # dofitems <> items orientation relationship
    xItem2SuperSetItems::Array{Union{Nothing,Vector{Int32}},1}  # only necessary for assembly of discontinuous operators ON_BFACES
end

function update_dii4op!(AM, j, DII::Type{<:DIIType}, item)
    @error "Do not know what to do with operator $j of type $DII"
end

# easiest case: continuously evaluatable operator
function update_dii4op!(AM::AssemblyManager, op::Int, ::Type{DIIType_continuous}, item::Int)
    AM.dofitems[op][1] = item
    if length(AM.EG) > 1
        # find EG index for geometry
        for j=1:length(AM.EG)
            if AM.xItemGeometries[item] == AM.EG[j]
                AM.EG4dofitem[op][1] = j
                break
            end
        end
    end
end

# jump operators on faces that are avluated using the cell basis
function update_dii4op!(AM::AssemblyManager, op::Int, ::Type{DIIType_discontinuous{DiscType,AT,basisAT}}, item::Int) where {DiscType, AT <: ON_FACES, basisAT <: ON_CELLS}
    AM.dofitems[op][1] = AM.xDofItems4Item[op][1,item]
    AM.dofitems[op][2] = AM.xDofItems4Item[op][2,item]
    for k = 1 : num_targets(AM.xItemInDofItems[op],AM.dofitems[op][1])
        if AM.xItemInDofItems[op][k,AM.dofitems[op][1]] == item
            AM.itempos4dofitem[op][1] = k
            AM.orientation4dofitem[op][1] = AM.xDofItemItemOrientations[op][k, AM.dofitems[op][1]]
            break
        end
    end
    if AM.dofitems[op][2] > 0
        for k = 1 : num_targets(AM.xItemInDofItems[op],AM.dofitems[op][2])
            if AM.xItemInDofItems[op][k,AM.dofitems[op][2]] == item
                AM.itempos4dofitem[op][2] = k
                AM.orientation4dofitem[op][2] = AM.xDofItemItemOrientations[op][k, AM.dofitems[op][2]]
                break
            end
        end
        if DiscType == Jump
            AM.coeff4dofitem[op][1] = 1
            AM.coeff4dofitem[op][2] = -1
        elseif DiscType == Average
            AM.coeff4dofitem[op][1] = 0.5
            AM.coeff4dofitem[op][2] = 0.5
        end
    else
        AM.coeff4dofitem[op][1] = 1
        AM.coeff4dofitem[op][2] = 0
        if AT == ON_IFACES
            # if assembly is only on interior faces, ignore boundary faces by setting dofitems to zero
            AM.dofitems[op][1] = 0
        end
    end

    if length(AM.dEG) > 1
        # find EG index for geometry
        if AM.dofitems[op][1] > 0
            for j=1:length(AM.dEG)
                if AM.xDofItemGeometries[op][AM.dofitems[op][1]] == AM.dEG[j]
                    AM.EG4dofitem[op][1] = length(AM.EG) + j
                    break
                end
            end
        end
        if AM.dofitems[op][2] > 0
            for j=1:length(AM.dEG)
                if AM.xDofItemGeometries[op][AM.dofitems[op][2]] == AM.dEG[j]
                    AM.EG4dofitem[op][2] = length(AM.EG) + j
                    break
                end
            end
        end
    else
        AM.EG4dofitem[op][1] = length(AM.EG) + 1
        AM.EG4dofitem[op][2] = length(AM.EG) + 1
    end
end


# jump operators on bfaces that are avluated using the cell basis
function update_dii4op!(AM::AssemblyManager, op::Int, ::Type{DIIType_discontinuous{DiscType,AT,basisAT}}, item::Int) where {DiscType, AT <: ON_BFACES, basisAT <: ON_CELLS}

    AM.dofitems[op][1] = AM.xDofItems4Item[op][1,AM.xItem2SuperSetItems[op][item]]
    AM.dofitems[op][2] = 0
    for k = 1 : num_targets(AM.xItemInDofItems[op],AM.dofitems[op][1])
        if AM.xItemInDofItems[op][k,AM.dofitems[op][1]] == AM.dofitems[op][1]
            AM.itempos4dofitem[op][1] = k
            AM.orientation4dofitem[op][1] = AM.xDofItemItemOrientations[op][k, AM.dofitems[op][1]]
            break
        end
    end
    AM.coeff4dofitem[op][1] = 1
    AM.coeff4dofitem[op][2] = 0
    if length(AM.dEG) > 1
        # find EG index for geometry
        for j=1:length(AM.dEG)
            if AM.xDofItemGeometries[item] == AM.dEG[j]
                AM.EG4dofitem[op][1] = length(EG) + j
                break
            end
        end
    else
        AM.EG4dofitem[op][1] = length(AM.EG) + 1
    end
end



# broken space discontinuous operator on faces
function update_dii4op!(AM::AssemblyManager, op::Int, ::Type{DIIType_broken{DiscType,AT,basisAT}}, item::Int) where {DiscType, AT <: ON_FACES, basisAT <: ON_FACES}
    if AM.xDofItems4Item[op][2,item] > 0
        if DiscType == Jump
            AM.coeff4dofitem[op][1] = 1
            AM.coeff4dofitem[op][2] = -1
        elseif DiscType == Average
            AM.coeff4dofitem[op][1] = 0.5
            AM.coeff4dofitem[op][2] = 0.5
        end
        AM.dofitems[op][1] = item
        AM.dofitems[op][2] = item
    else
        if AT == ON_IFACES
            # if assembly is only on interior faces, ignore boundary faces by setting dofitems to zero
            AM.dofitems[op][1] = 0
        else
            AM.coeff4dofitem[op][1] = 1
            AM.coeff4dofitem[op][2] = 0
            AM.dofitems[op][1] = item
        end
        AM.dofitems[op][2] = 0
    end
    if length(AM.EG) > 1
        # find EG index for geometry
        for j=1:length(AM.EG)
            if AM.xItemGeometries[item] == AM.EG[j]
                AM.EG4dofitem[op][1] = j
                AM.EG4dofitem[op][2] = j
                AM.dofoffset4dofitem[op][2] = AM.ndofs4EG[op][j]
                break
            end
        end
    else
        AM.dofoffset4dofitem[op][2] = AM.ndofs4EG[op][1] 
    end
end

function update!(AM::AssemblyManager, item::Int)
    # get dofitem informations
    if AM.citem[] != item
        AM.citem[] = item
        for j = 1 : AM.nop
            # update dofitem information for assembly of operator
            # would like to avoid the if but dispatching by AM.dii4op_types[j] seems to cause allocations
            # so this avoids them at least for continuous operators
            if AM.dii4op_types[j] <: DIIType_continuous
                update_dii4op!(AM, j, DIIType_continuous, item)
            else
                update_dii4op!(AM, j, AM.dii4op_types[j], item)
            end
            # update needed basisevaler
            for di = 1 : length(AM.dofitems[j])
                if AM.dofitems[j][di] != 0
                    update!(AM.basisevaler[AM.EG4dofitem[j][di],j,AM.itempos4dofitem[j][di],AM.orientation4dofitem[j][di]], AM.dofitems[j][di])
                end
            end
        end
    end
end

@inline get_basisevaler(AM::AssemblyManager, op::Int, dofitem) = AM.basisevaler[AM.EG4dofitem[op][dofitem],op,AM.itempos4dofitem[op][dofitem],AM.orientation4dofitem[op][dofitem]]
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

"""
$(TYPEDEF)

each assembly pattern has one of the assembly pattern types (APT) that trigger different assemblies for the involved
finite element spaces, operators and an assigned action. The assembly type (AT) determines if the assembly takes
place on cells, faces or edges etc. (relatively to the assembly type of the first argument of the pattern)
"""
mutable struct AssemblyPattern{APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType, ActionType <: AbstractAction}
    name::String
    FES::Array{FESpace,1}
    operators::Array{DataType,1}
    newton_args::Array{Int,1}      # which of the operators should be differentiated in this block ? (only relevant for nonlinear form)
    action::ActionType
    apply_action_to::Array{Int,1}
    regions::Array{Int,1}
    AM::AssemblyManager{T} # hidden stuff needed for assembly
    AssemblyPattern() = new{APT_Undefined, Float64, ON_CELLS, NoAction}()
    AssemblyPattern{APT, T, AT}() where {APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType} = new{APT,T,AT,NoAction}()
    AssemblyPattern{APT, T, AT}(name,FES,operators,action,apply_to,regions) where {APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType}  = new{APT,T,AT,typeof(action)}(name, FES,operators,[],action,apply_to,regions)
    AssemblyPattern{APT, T, AT}(FES,operators,action,apply_to,regions) where {APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType}  = new{APT,T,AT,typeof(action)}("$APT", FES,operators,[],action,apply_to,regions)
end 

function Base.show(io::IO, AP::AssemblyPattern)
    println(io,"\n\tpattern name = $(AP.name)")
	println(io,"\tpattern type = $(typeof(AP).parameters[1]) (T = $(typeof(AP).parameters[2]))")
    println(io,"\tpattern span = $(typeof(AP).parameters[3]) (regions = $(AP.regions))")
    println(io,"\tpattern oprs = $(AP.operators)")
    if !(typeof(AP.action) <: NoAction)
        println(io,"\tpattern actn = $(AP.action.name) (apply_to = $(AP.apply_action_to) size = $(AP.action.argsizes))")
    end
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
        # if AT == ON_EDGES: todo (should lead to ON_CELL assembly on neighbouring CELLS, more than two)
        DofitemInformation4Operator(FES, AT, basisAT, FO.parameters[posdt])
    else
        # call standard handlers
        # assembly as specified by AT
        DofitemInformation4Operator(FES, AT, basisAT)
    end
end

# default handlers for continuous operators
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:AbstractAssemblyType}, basisAT::Type{<:AbstractAssemblyType})
    xgrid = FES.xgrid
    xItemGeometries = xgrid[GridComponentGeometries4AssemblyType(AT)]
    return DIIType_continuous, xItemGeometries, nothing, nothing, nothing, nothing
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
    return DIIType_discontinuous{DiscType,AT,basisAT}, xCellGeometries, xFaceCells, xCellFaces, xCellFaceOrientations, nothing
end


# special handlers for jump operators on FACES of broken spaces that otherwise have the necessary continuity (so that they can use the ON_FACES basis)
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_FACES}, basisAT::Type{<:ON_FACES}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xItemGeometries = xgrid[FaceGeometries]
    return DIIType_broken{DiscType,AT,basisAT}, xItemGeometries, xFaceCells, nothing, nothing, nothing
end


# special handlers for jump operators with ON_CELL basis
function DofitemInformation4Operator(FES::FESpace, AT::Type{<:ON_BFACES}, basisAT::Type{<:ON_CELLS}, DiscType::Type{<:Union{Jump, Average}})
    xgrid = FES.xgrid
    xFaceCells = xgrid[FaceCells]
    xCellFaces = xgrid[CellFaces]
    xCellGeometries = xgrid[CellGeometries]
    xBFaces = xgrid[BFaces]
    xCellFaceOrientations = xgrid[CellFaceOrientations]
    return DIIType_discontinuous{DiscType,AT,basisAT}, xCellGeometries, xFaceCells, xCellFaces, xCellFaceOrientations, xBFaces
end





function prepare_assembly!(AP::AssemblyPattern{APT,T,AT}; FES = "from AP") where {APT <: AssemblyPatternType, T<: Real, AT <: AbstractAssemblyType}

    if FES != "from AP"
        FE = FES
    else
        FE = AP.FES
    end
    bonus_quadorder::Int = AP.action.bonus_quadorder[]
    operator = AP.operators
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
    # dii4op = Array{Function,1}(undef,length(FE))
    for j=1:length(FE)
        dofitemAT[j] = DefaultBasisAssemblyType4Operator(operator[j], AT, eltype(FE[j]))
        xItemDofs[j] = Dofmap4AssemblyType(FE[j],dofitemAT[j])

        #broken_space = FE[j].broken
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
            if !(k in discontinuous_operators)
                if k > 1 && FE[k] == FE[1] && operator[k] == operator[1]
                    basisevaler[j,k,1,1] = basisevaler[j,1,1,1] # e.g. for symmetric bilinearforms
                elseif k > 2 && FE[k] == FE[2] && operator[k] == operator[2]
                    basisevaler[j,k,1,1] = basisevaler[j,2,1,1]
                else    
                    basisevaler[j,k,1,1] = FEBasisEvaluator{T,eltype(FE[k]),EG[j],operator[k],AT}(FE[k], qf[j])
                end    
                ndofs4EG[k][j] = size(basisevaler[j,k,1,1].cvals,2)
            else
                # todo: will not be evaluated, but do something reasonable here
                basisevaler[j,k,1,1] = FEBasisEvaluator{T,eltype(FE[k]),EG[j],Identity,AT}(FE[k], qf[j])
                ndofs4EG[k][j] = 0
            end
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
                    basisevaler[EGoffset + j,k,f,orientation] = FEBasisEvaluator{T,eltype(FE[k]),EGdofitem[j],operator[k],dofitemAT[k]}(FE[k], qf[EGoffset + j])
                end
                ndofs4EG[k][EGoffset+j] = size(basisevaler[EGoffset + j,k,1,1].cvals,2)
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
    xDofItemGeometries = Array{GridEGTypes,1}(undef, length(FE))           
    xDofItems4Item = Array{Union{Nothing,GridAdjacencyTypes{Int32}},1}(undef, length(FE))     
    xItemInDofItems = Array{Union{Nothing,GridAdjacencyTypes{Int32}},1}(undef, length(FE))     
    xDofItemItemOrientations = Array{Union{Nothing,GridAdjacencyTypes{Int32}},1}(undef, length(FE))
    xItem2SuperSetItems = Array{Union{Nothing,Vector{Int32}},1}(undef, length(FE))
    dii4op_types = Array{Type{<:DIIType},1}(undef,length(FE))

    for j = 1 : length(FE)
        dofitems[j] = (j in continuous_operators) ? zeros(Int,1) : zeros(Int,2)
        EG4dofitem[j] = [1,1]
        itempos4dofitem[j] = [1,1]
        coeff4dofitem[j] = ones(T,2)
        dofoffset4dofitem[j] = zeros(Int,2)
        orientation4dofitem[j] = [1,1]
        xItemDofs[j] = Dofmap4AssemblyType(FE[j], dofitemAT[j])
        b,c,d,e,f,g = DofitemInformation4Operator(FE[j], AT, dofitemAT[j], operator[j])
        dii4op_types[j] = b
        xDofItemGeometries[j] = c
        xDofItems4Item[j] = d
        xItemInDofItems[j] = e
        xDofItemItemOrientations[j] = f
        xItem2SuperSetItems[j] = g
    end

    xItemGeometries = FE[1].xgrid[GridComponentGeometries4AssemblyType(AT)]
    citem = 0
    AP.AM = AssemblyManager{T}(xItemDofs,ndofs4EG,length(FE),qf,basisevaler,dii4op_types,dofitemAT,Ref(citem),dofitems,EG4dofitem,itempos4dofitem,coeff4dofitem,dofoffset4dofitem,orientation4dofitem,EG,EGdofitem,xItemGeometries,xDofItemGeometries,xDofItems4Item,xItemInDofItems,xDofItemItemOrientations,xItem2SuperSetItems)
end

# each assembly pattern is in its own file
include("assemblypatterns/pointevaluator.jl")
include("assemblypatterns/itemintegrator.jl")
include("assemblypatterns/segmentintegrator.jl")
include("assemblypatterns/linearform.jl")
include("assemblypatterns/bilinearform.jl")
include("assemblypatterns/trilinearform.jl")
include("assemblypatterns/multilinearform.jl")
include("assemblypatterns/nonlinearform.jl")