
using ExtendableSparse

function test_qpmatchup(xgrid)

    AT = ON_IFACES
    ncomponents = 1
    FE = FESpace{H1P0{ncomponents}}(xgrid)

    AP = DiscreteLinearForm([GradientDisc{Jump}], [FE]; AT = AT)

    # prepare assembly
    prepare_assembly!(AP)
    AM::AssemblyManager = AP.AM
    nitems::Int = num_sources(xgrid[CellNodes])
    maxerror = 0.0
    pointerror = 0.0
    xleft = [0.0,0.0,0.0]
    xright = [0.0,0.0,0.0]
    for item = 1 : nitems

        # get dofitem informations
        GradientRobustMultiPhysics.update_assembly!(AM, item)

        # check that quadrature points on both sides match
        if AM.dofitems[1][2] > 0
            bleft = AM.basisevaler[AM.EG4dofitem[1][1],1,AM.itempos4dofitem[1][1],AM.orientation4dofitem[1][1]]
            bright = AM.basisevaler[AM.EG4dofitem[1][2],1,AM.itempos4dofitem[1][2],AM.orientation4dofitem[1][2]]
            GradientRobustMultiPhysics.update_trafo!(bleft.L2G,AM.dofitems[1][1])
            GradientRobustMultiPhysics.update_trafo!(bright.L2G,AM.dofitems[1][2])
            for i = 1 : length(bleft.xref)
                eval_trafo!(xleft, bleft.L2G, bleft.xref[i])
                eval_trafo!(xright, bright.L2G, bright.xref[i])
                for k = 1 : 3
                    pointerror += (xleft[k] - xright[k])^2
                end
                if pointerror > 1e-14
                    @error "orientations = $orientation4dofitem xleft = $xleft right = $xright error = $pointerror"
                end
                maxerror = max(maxerror, pointerror)
            end
        end
    end

    return maxerror
end


function test_disc_LF(xgrid, discontinuity)

    # generate constant P0 function with no jumps
    ncomponents = 1
    FE = FESpace{H1P1{ncomponents}}(xgrid)
    FEFunction = FEVector{Float64}("velocity",FE)
    fill!(FEFunction[1],1)


    if discontinuity == Parent
        # test Parent{1} - Parent{2} version of jump
        action = Action((result, input) -> (result[1] = input[1] - input[2]), [1,2]; bonus_quadorder = 1)
        TestIntegrator = ItemIntegrator(Float64, ON_IFACES, [Parent{1}(Identity), Parent{2}(Identity)], action)
        error = evaluate(TestIntegrator, [FEFunction[1], FEFunction[1]])
    else
        action = NoAction()
        TestForm = DiscreteLinearForm([IdentityDisc{discontinuity}], [FE], action; AT = ON_IFACES)
        b = zeros(Float64,FE.ndofs)
        assemble!(b, TestForm)
        error = 0
        for j = 1 : FE.ndofs
                error += b[j] * FEFunction[1][j]
        end
        if discontinuity == Average
            for j = 1 : num_sources(xgrid[FaceNodes])
                if xgrid[FaceCells][2,j] != 0
                    error -= xgrid[FaceVolumes][j]
                end
            end
        end
    end

    return error
end

function test_disc_BLF(xgrid, discontinuity)

    # generate constant P0 function with no jumps
    ncomponents = 1
    FE = FESpace{H1P0{ncomponents}}(xgrid)
    FEFunction = FEVector{Float64}("velocity",FE)
    fill!(FEFunction[1],1)
 
    action = NoAction()
    TestForm = DiscreteBilinearForm([IdentityDisc{discontinuity}, IdentityDisc{discontinuity}], [FE, FE], action; AT = ON_IFACES)
    A = FEMatrix{Float64}("test",FE)
    assemble!(A[1,1], TestForm)
    flush!(A[1,1].entries)

    # average should equal length of interior skeleton
    error = zeros(Float64,3)
    error[1] = lrmatmul(FEFunction[1].entries, A[1,1].entries, FEFunction[1].entries; factor = 1)
    if discontinuity == Average
        for j = 1 : num_sources(xgrid[FaceNodes])
            if xgrid[FaceCells][2,j] != 0
                error[1] -= xgrid[FaceVolumes][j]
            end
        end
    end
    
    return error
 end