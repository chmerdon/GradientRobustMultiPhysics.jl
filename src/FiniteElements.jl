module FiniteElements

using ExtendableGrids
using ExtendableSparse
using FEXGrid
using QuadratureRules
using ForwardDiff # for FEBasisEvaluator


 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################


abstract type AbstractFiniteElement end

  # subtype for Hdiv-conforming elements
  abstract type AbstractHdivFiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/Hdiv_RT0.jl");

    # second order
    #include("FEdefinitions/HDIV_RT1.jl");
    #include("FEdefinitions/HDIV_BDM1.jl");

  # subtype for H1 conforming elements (also Crouzeix-Raviart)
  abstract type AbstractH1FiniteElement <: AbstractFiniteElement end
    # lowest order
    include("FEdefinitions/H1_P1.jl");
    #include("FEdefinitions/H1_Q1.jl");
    #include("FEdefinitions/H1_MINI.jl");
    #include("FEdefinitions/H1_CR.jl");
    # second order
    include("FEdefinitions/H1_P2.jl");
    #include("FEdefinitions/H1_P2B.jl");

    abstract type AbstractH1FiniteElementWithCoefficients <: AbstractH1FiniteElement end
      include("FEdefinitions/H1v_BR.jl");

 
  # subtype for L2 conforming elements
  abstract type AbstractL2FiniteElement <: AbstractFiniteElement end
    include("FEdefinitions/L2_P0.jl"); # currently masked as H1 element
    #include("FEdefinitions/L2_P1.jl");
 
  # subtype for Hcurl-conforming elements
  abstract type AbstractHcurlFiniteElement <: AbstractFiniteElement end
    # TODO

export AbstractFiniteElement, AbstractH1FiniteElementWithCoefficients, AbstractH1FiniteElement, AbstractHdivFiniteElement, AbstractHcurlFiniteElement
export get_ncomponents
export getH1P1FiniteElement, getH1BRFiniteElement, getH1P2FiniteElement
export getHdivRT0FiniteElement
export interpolate!, nodevalues!

# show function for FiniteElement
function show(FE::AbstractFiniteElement)
	println("\nFiniteElement information")
	println("=========================")
	println("   name = " * FE.name)
	println("  ndofs = $(FE.ndofs)")
end


include("FiniteElements_FEBlockArrays.jl");
export FEVectorBlock, FEVector, FEMatrix, FEMatrixBlock


include("FiniteElements_FEBasisEvaluator.jl")
export FEBasisEvaluator, update!
export AbstractFunctionOperator
export Identity, Gradient, SymmetricGradient, Laplacian, Hessian, Curl, Rotation, Divergence, Trace, Deviator
export NeededDerivatives4Operator, QuadratureOrderShift4Operator, FEPropertyDofs4AssemblyType










function interpolate!(Target::FEVectorBlock, exact_function!::Function; dofs = [], verbosity::Int = 0, bonus_quadorder::Int = 0)
    if verbosity > 0
        println("\nINTERPOLATING")
        println("=============")
        println("     target = $(Target.name)")
        println("         FE = $(Target.FEType.name) (ndofs = $(Target.FEType.ndofs))")
    end
    interpolate!(Target, Target.FEType, exact_function!; dofs = dofs, bonus_quadorder = bonus_quadorder)
end


# abstract nodevalue function that works for any element and can be overwritten for special ones
# (e.g. P1/P2 etc. just have to return their dofs, see H1_P1.jl)
function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::AbstractFiniteElement; regions::Array{Int,1} = [0])
  xItemGeometries = FE.xgrid[CellGeometries]
  xItemRegions = FE.xgrid[CellRegions]
  xItemDofs = FE.CellDofs
  xItemNodes = FE.xgrid[CellNodes]

  T = eltype(Target)
  if regions == [0]
      try
          regions = Array{Int32,1}(Base.unique(xItemRegions[:]))
      catch
          regions = [xItemRegions[1]]
      end        
  else
      regions = Array{Int32,1}(regions)    
  end

  EG, ndofs4EG = uniqueEG(xItemGeometries, xItemRegions, [xItemDofs], regions)
  qf = Array{QuadratureRule,1}(undef,length(EG))
  basisevaler = Array{FEBasisEvaluator,1}(undef,length(EG))
  for j = 1 : length(EG)
      qf[j] = VertexRule(EG[j])
      basisevaler[j] = FEBasisEvaluator{T,typeof(FE),EG[j],Identity,AbstractAssemblyTypeCELL}(FE, qf[j])
  end    

  nitems::Int = num_sources(xItemDofs)
  nnodes::Int = num_sources(FE.xgrid[Coordinates])
  nneighbours = zeros(Int,nnodes)
  dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
  basisvals::Array{T,3} = basisevaler[1].cvals # pointer to operator results
  item::Int = 0
  nregions::Int = length(regions)
  ncomponents::Int = get_ncomponents(typeof(FE))
  iEG::Int = 0
  for item = 1 : nitems
    for r = 1 : nregions
    # check if item region is in regions
    if xItemRegions[item] == regions[r]

        # find index for CellType
        itemET = xItemGeometries[item]
        for j=1:length(EG)
            if itemET == EG[j]
                iEG = j
                break;
            end
        end
        ndofs4item = ndofs4EG[1][iEG]

        # update FEbasisevaler
        FiniteElements.update!(basisevaler[iEG],item)
        basisvals = basisevaler[iEG].cvals

        # update dofs
        for dof_i=1:ndofs4item
            dofs[dof_i] = xItemDofs[dof_i,item]
        end

        for i in eachindex(qf[iEG].w) # vertices
            node = xItemNodes[i,item]
            nneighbours[node] += 1
            for k = 1 : ncomponents
                for dof_i=1:ndofs4item
                  Target[k,node] += Source[dofs[dof_i]] * basisvals[k,dof_i,i]
                end
            end  
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

    for node = 1 : nnodes, k = 1 : ncomponents
      Target[k,node] /= nneighbours[node]
    end
end



end #module
