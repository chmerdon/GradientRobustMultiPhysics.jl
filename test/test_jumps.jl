
using ExtendableSparse

function test_qpmatchup(xgrid)

    AT = ON_IFACES
    ncomponents = 1
    FE = FESpace{H1P0{ncomponents}}(xgrid)
    T = Float64

    action = NoAction()
    AP = LinearForm(Float64, AT, [FE], [GradientDisc{Jump}], action)

    # prepare assembly
    prepare_assembly!(AP)
    ndofs4EG = AP.AM.ndofs4EG
    qf = AP.AM.qf
    basisevaler = AP.AM.basisevaler
    dii4op = AP.AM.dii4op

    operators = AP.operators
    xItemNodes = FE.xgrid[GridComponentNodes4AssemblyType(AT)]
    xItemVolumes::Array{T,1} = FE.xgrid[GridComponentVolumes4AssemblyType(AT)]
    xItemDofs::Union{VariableTargetAdjacency{Int32},SerialVariableTargetAdjacency{Int32},Array{Int32,2}} = Dofmap4AssemblyType(FE, AP.AM.basisAT[1])
    nitems = Int64(num_sources(xItemNodes))

 
    EG4item = 0
    EG4dofitem = [1,1] # type of the current item
    ndofs4dofitem = 0 # number of dofs for item
    dofitems = [0,0] # itemnr where the dof numbers can be found
    itempos4dofitem::Array{Int,1} = [1,1] # local item position in dofitem
    orientation4dofitem::Array{Int,1} = [1,2] # local orientation
    dofoffset4dofitem::Array{Int,1} = [1,2] # local orientation
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
        EG4item = dii4op[1](dofitems, EG4dofitem, itempos4dofitem, coefficient4dofitem, orientation4dofitem, dofoffset4dofitem, item)

        # check that quadrature points on both sides match
        if dofitems[2] > 0
            bleft = basisevaler[EG4dofitem[1],1,itempos4dofitem[1],orientation4dofitem[1]]
            bright = basisevaler[EG4dofitem[2],1,itempos4dofitem[2],orientation4dofitem[2]]
            GradientRobustMultiPhysics.update!(bleft.L2G,dofitems[1])
            GradientRobustMultiPhysics.update!(bright.L2G,dofitems[2])
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


function test_disc_LF(xgrid, discontinuity)

    # generate constant P0 function with no jumps
    ncomponents = 1
    FE = FESpace{H1P1{ncomponents}}(xgrid)
    FEFunction = FEVector{Float64}("velocity",FE)
    fill!(FEFunction[1],1)

    action = NoAction()
    TestForm = LinearForm(Float64,ON_IFACES, [FE], [IdentityDisc{discontinuity}], action)
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
        assemble!(b,FEFunction[1], TestForm; fixed_argument = c)
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
    action = Action(Float64, ActionKernel(action_kernel, [1,2]; quadorder = 0))
    TestForm = TrilinearForm(Float64, ON_IFACES, Array{FESpace,1}([FE, FE, FE]), [IdentityDisc{discontinuity}, IdentityDisc{Average}, IdentityDisc{Average}], action)

    # average should equal length of interior skeleton
    error = zeros(Float64,4)
    for c = 1 : 3
        A = FEMatrix{Float64}("test",FE)
        assemble!(A[1,1], FEFunction[1], TestForm; fixed_argument = c)
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
    assemble!(b,FEFunction[1],FEFunction[1], TestForm)
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