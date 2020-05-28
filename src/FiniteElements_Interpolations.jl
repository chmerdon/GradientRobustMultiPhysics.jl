function interpolate!(Target::FEVectorBlock, exact_function!::Function; dofs = [], verbosity::Int = 0, bonus_quadorder::Int = 0)
    if verbosity > 0
        println("\nINTERPOLATING")
        println("=============")
        println("     target = $(Target.name)")
        println("         FE = $(Target.FES.name) (ndofs = $(Target.FES.ndofs))")
    end
    interpolate!(Target, Target.FES, exact_function!; dofs = dofs, bonus_quadorder = bonus_quadorder)
end


# abstract nodevalue function that works for any element and can be overwritten for special ones
function nodevalues!(Target::AbstractArray{<:Real,2}, Source::AbstractArray{<:Real,1}, FE::FESpace; regions::Array{Int,1} = [0])
  xItemGeometries = FE.xgrid[CellGeometries]
  xItemRegions = FE.xgrid[CellRegions]
  xItemDofs = FE.CellDofs
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
  FEType = Base.eltype(typeof(FE))
  for j = 1 : length(EG)
      qf[j] = VertexRule(EG[j])
      basisevaler[j] = FEBasisEvaluator{T,FEType,EG[j],Identity,AbstractAssemblyTypeCELL}(FE, qf[j])
  end    

  nitems::Int = num_sources(xItemDofs)
  nnodes::Int = num_sources(FE.xgrid[Coordinates])
  nneighbours = zeros(Int,nnodes)
  dofs = zeros(Int32,max_num_targets_per_source(xItemDofs))
  basisvals::Array{T,3} = basisevaler[1].cvals # pointer to operator results
  item::Int = 0
  nregions::Int = length(regions)
  ncomponents::Int = get_ncomponents(FEType)
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