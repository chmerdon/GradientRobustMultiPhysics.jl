function gridgen_unitinterval(maxarea)
    grid = Grid.Mesh{Float64}(Array{Float64,2}(Array{Float64,2}([0,0.5,1]')'),Array{Int64,2}([1 2;2 3]),Grid.ElemType1DInterval(),ceil(log2(1/maxarea)));
    Grid.ensure_bfaces!(grid)
    return grid;
end