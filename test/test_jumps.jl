
function test_jumps(xgrid; verbosity = 0)

    AT = ON_IFACES
    ncomponents = 1
    FE = FESpace{H1P1{ncomponents}}(xgrid)
    T = Float64

    action = DoNotChangeAction(ncomponents)
    TestForm = LinearForm{Float64,AT}(FE, IdentityDisc{Jump}, action, [0])

    operator = TestForm.operator
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{T,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE, DofitemAT4Operator(AT, operator))
    nitems = Int64(num_sources(xItemNodes))

    # prepare assembly
    regions::Array{Int32,1} = [1]
    bonus_quadorder = 5
    EG, ndofs4EG, qf, basisevaler, dii4op =  GradientRobustMultiPhysics.prepare_assembly(TestForm, [operator], [FE], regions, 1, bonus_quadorder, verbosity - 1)
 
    EG4item = 0
    EG4dofitem = [1,1] # type of the current item
    ndofs4dofitem = 0 # number of dofs for item
    dofitems = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,2] # local orientation
    coefficient4dofitem::Array{T,1} = [0.0,0.0]
    dofitem = 0
    weights::Array{T,1} = qf[1].w # somehow this saves A LOT allocations
    basisevaler4dofitem::FEBasisEvaluator = basisevaler[1,1,1,1]
    maxerror = 0.0
    pointerror = 0.0
    xleft = [0.0,0.0,0.0]
    xright = [0.0,0.0,0.0]
    for item = 1 : nitems

        # get dofitem informations
        EG4item = dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, item)

        # check that quadrature points on both sides match
        if dofitems[2] > 0
            bleft = basisevaler[EG4dofitem[1],1,itempos4dofitem[1],orientation4dofitem[1]]
            bright = basisevaler[EG4dofitem[2],1,itempos4dofitem[2],orientation4dofitem[2]]
            update!(bleft.L2G,dofitems[1])
            update!(bright.L2G,dofitems[2])
            for i = 1 : length(bleft.xref)
                eval!(xleft, bleft.L2G, bleft.xref[i])
                eval!(xright, bright.L2G, bright.xref[i])
                for k = 1 : 3
                    pointerror += (xleft[k] - xright[k])^2
                end
                if pointerror > 1e-14
                    println("ERROR")
                    println("orientations = $orientation4dofitem xleft = $xleft right = $xright error = $pointerror")
                end
                maxerror = max(maxerror, pointerror)
            end
        end
    end

    return maxerror
end