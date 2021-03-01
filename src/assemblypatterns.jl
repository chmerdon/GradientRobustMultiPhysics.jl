############################
# AbstractAssemblyPatterns #
############################

abstract type AssemblyPatternType end
abstract type APT_Undefined <: AssemblyPatternType end

# backpack to save all the information needed for assembly
# like element geometries (EG), quadrature formulas (qf), basisevaluators for the FES, ...
# idea is to store this to avoid recomputation in e.g. in iterative schemes
mutable struct AssemblyPatternPreparations
    EG
    ndofs4EG
    qf
    basisevaler
    dii4op
    basisAT
end

struct AssemblyPattern{APT <: AssemblyPatternType, T <: Real, AT <: AbstractAssemblyType}
    FES::Array{FESpace,1}
    operators::Array{DataType,1}
    action::AbstractAction
    regions::Array{Int,1}
    APP::AssemblyPatternPreparations # hidden stuff needed for assembly
end 

function EmptyAssemblyPattern()
    return AssemblyPattern{APT_Undefined, Float64, ON_CELLS}([],[],DoNotChangeAction(1),[0],AssemblyPatternPreparations(nothing,nothing,nothing,nothing,nothing,nothing))
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
            #push!(broken_operators,j)
            push!(discontinuous_operators,j)
            dofitemAT[j] = ON_CELLS
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

            # generate new quadrature rules on neighbouring cells
            # where quadrature points of face are mapped to quadrature points of cells
            qf[EGoffset + j] = QuadratureRule{T,EGdofitem[j]}(qf4face.name * " (shape faces)",Array{Array{T,1},1}(undef,length(qf4face.xref)),qf4face.w)
            for k in discontinuous_operators
                xrefFACE2CELL = xrefFACE2xrefCELL(EGdofitem[j])
                EGface = facetype_of_cellface(EGdofitem[j], 1)
                xrefFACE2OFACE = xrefFACE2xrefOFACE(EGface)
                for f = 1 : nfaces4cell, orientation = 1 : length(xrefFACE2OFACE)
                    for i = 1 : length(qf4face.xref)
                        qf[EGoffset + j].xref[i] = xrefFACE2CELL[f](xrefFACE2OFACE[orientation](qf4face.xref[i]))
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
            println("      ($k) $(FE[k].name) / $(operator[k]) / $(ndofs4EG[k]) / $(basisAT[k])")
        end    
        if verbosity > 1
            for j = 1 : length(EG)
                println("\nQuadratureRule [$j] for $(EG[j]):")
                Base.show(qf[j])
            end
        end
    end

    # write down assembly preparations
    AP.APP.EG = EG
    AP.APP.ndofs4EG = ndofs4EG
    AP.APP.qf = qf
    AP.APP.basisevaler = basisevaler
    AP.APP.dii4op = dii4op
    AP.APP.basisAT = dofitemAT
end

# each assembly pattern is in its own file
include("assemblypatterns/itemintegrator.jl")
include("assemblypatterns/linearform.jl")
include("assemblypatterns/bilinearform.jl")
include("assemblypatterns/trilinearform.jl")
include("assemblypatterns/multilinearform.jl")
include("assemblypatterns/nonlinearform.jl")