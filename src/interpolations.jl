
# this method calls the interpolate! method specified by the finite element if available
function interpolate!(Target::FEVectorBlock, source_function!::Function; dofs = [], verbosity::Int = 0, bonus_quadorder::Int = 0, time = 0)
    if verbosity > 0
        println("\nINTERPOLATING")
        println("=============")
        println("     target = $(Target.name)")
        println("         FE = $(Target.FES.name) (ndofs = $(Target.FES.ndofs))")
    end
    # check if function is time-dependent
    if applicable(source_function!,[0],0,0)
        source_function_fixt!(result,x) = source_function!(result,x,time)
        interpolate!(Target, Target.FES, source_function_fixt!; dofs = dofs, bonus_quadorder = bonus_quadorder)
    else
        interpolate!(Target, Target.FES, source_function!; dofs = dofs, bonus_quadorder = bonus_quadorder)
    end
end


"""
$(TYPEDSIGNATURES)

Evaluates the finite element function with the specified coefficient vector Source (interpeted as an element of the finite element space FE)
and the specified FunctionOperator at all the nodes of the grids and writes them into Target. Discontinuous quantities are averaged.
"""
function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace, operator::Type{<:AbstractFunctionOperator} = Identity; regions::Array{Int,1} = [0], continuous::Bool = false)
  xItemGeometries = FE.xgrid[CellGeometries]
  xItemRegions = FE.xgrid[CellRegions]
  xItemDofs = FE.dofmaps[CellDofs]
  xItemNodes = FE.xgrid[CellNodes]

  T = Base.eltype(Target)
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
  FEType = Base.eltype(FE)
  for j = 1 : length(EG)
      qf[j] = VertexRule(EG[j])
      basisevaler[j] = FEBasisEvaluator{T,FEType,EG[j],operator,ON_CELLS}(FE, qf[j])
  end    
  cvals_resultdim::Int = size(basisevaler[1].cvals,1)
  @assert size(Target,1) >= cvals_resultdim

  nitems::Int = num_sources(xItemDofs)
  nnodes::Int = num_sources(FE.xgrid[Coordinates])
  nneighbours = zeros(Int,nnodes)
  dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
  basisvals::Array{T,3} = basisevaler[1].cvals # pointer to operator results
  item::Int = 0
  nregions::Int = length(regions)
  ncomponents::Int = get_ncomponents(FEType)
  iEG::Int = 0
  node::Int = 0
  dof::Int = 0
  flag4node = zeros(Bool,nnodes)
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

        # update FEbasisevaler
        update!(basisevaler[iEG],item)
        basisvals = basisevaler[iEG].cvals

        for i in eachindex(qf[iEG].w) # vertices
            node = xItemNodes[i,item]
            if continuous == false || flag4node[node] == false
                nneighbours[node] += 1
                for k = 1 :  size(basisvals,1)
                    for dof_i = 1 : ndofs4EG[1][iEG]
                        dof = xItemDofs[dof_i,item]
                        Target[k,node] += Source[dof] * basisvals[k,dof_i,i]
                    end
                end  
            end
        end  
        break; # region for loop
    end # if in region    
    end # region for loop
    end # item for loop

    if continuous == false
        for node = 1 : nnodes, k = 1 : cvals_resultdim
            Target[k,node] /= nneighbours[node]
        end
    end
end