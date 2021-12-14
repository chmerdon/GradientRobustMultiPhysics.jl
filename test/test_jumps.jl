
using ExtendableSparse

function test_qpmatchup(xgrid)

    AT = ON_IFACES
    ncomponents = 1
    FE = FESpace{H1P0{ncomponents}}(xgrid)
    T = Float64

    AP = LinearForm(Float64, AT, [FE], [GradientDisc{Jump}])

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
        action = Action((result, input) -> (result[1] = input[1] - input[2]), [1,2]; quadorder = 1)
        TestIntegrator = ItemIntegrator(Float64, ON_IFACES, [Parent{1}(Identity), Parent{2}(Identity)], action)
        error = evaluate(TestIntegrator, [FEFunction[1], FEFunction[1]])
    else
        action = NoAction()
        TestForm = LinearForm(Float64, ON_IFACES, [FE], [IdentityDisc{discontinuity}], action)
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
    TestForm = BilinearForm(Float64,ON_IFACES,[FE, FE], [IdentityDisc{discontinuity}, IdentityDisc{discontinuity}], action)
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

    # once again as a bilinear form where one component is fixed
    for c = 1 : 2
        b = zeros(Float64,FE.ndofs)
        assemble!(b, TestForm, FEFunction[1]; fixed_arguments = [c])
        for j = 1 : FE.ndofs
            error[1+c] += b[j] * FEFunction[1][j]
        end
        if discontinuity == Average
            for j = 1 : num_sources(xgrid[FaceNodes])
                if xgrid[FaceCells][2,j] != 0
                    error[1+c] -= xgrid[FaceVolumes][j]
                end
            end
        end
    end
    
    return error
 end


function test_disc_TLF(xgrid, discontinuity)

    # generate constant P0 function with no jumps
    ncomponents = 1
    FE = FESpace{H1P0{ncomponents}}(xgrid)
    FEFunction = FEVector{Float64}("velocity",FE)
    fill!(FEFunction[1],1)
 
    function action_kernel(result, input)
        # input = [DiscIdentity{discontinuity}, DiscIdentity{discontinuity}]
        result[1] = input[1] * input[2]
        return nothing
    end
    action = Action{Float64}( ActionKernel(action_kernel, [1,2]; quadorder = 0))
    TestForm = TrilinearForm(Float64, ON_IFACES, Array{FESpace{Float64,Int32},1}([FE, FE, FE]), [IdentityDisc{discontinuity}, IdentityDisc{Average}, IdentityDisc{Average}], action)

    # average should equal length of interior skeleton
    error = zeros(Float64,4)
    for c = 1 : 3
        A = FEMatrix{Float64}("test",FE)
        assemble!(A[1,1], TestForm, [FEFunction[1]]; fixed_arguments = [c])
        flush!(A[1,1].entries)
        error[c] = lrmatmul(FEFunction[1].entries, A[1,1].entries, FEFunction[1].entries; factor = 1)
        if discontinuity == Average
            for j = 1 : num_sources(xgrid[FaceNodes])
                if xgrid[FaceCells][2,j] != 0
                    error[c] -= xgrid[FaceVolumes][j]
                end
            end
        end
    end

    # once again as a trilinearform where the first two components are fixed
    b = zeros(Float64,FE.ndofs)
    assemble!(b, TestForm, [FEFunction[1],FEFunction[1]])
    for j = 1 : FE.ndofs
        error[4] += b[j] * FEFunction[1][j]
    end
    if discontinuity == Average
        for j = 1 : num_sources(xgrid[FaceNodes])
            if xgrid[FaceCells][2,j] != 0
                error[4] -= xgrid[FaceVolumes][j]
            end
        end
    end
    
    return error
 end
