using PyPlot
#using GLMakie
#using Plots
using ExtendableSparse
using ExtendableGrids
using GradientRobustMultiPhysics

function make_all(; Plotter = nothing)

    # generate images for all examples
    # (this is called only manually at the moment)
    example_jl_dir = joinpath(@__DIR__,"..","examples")
    example_img_dir  = joinpath(@__DIR__,"src","images")

    for example_source in readdir(example_jl_dir)
        base,ext=splitext(example_source)
        if example_source[1:7] == "Example" && ext==".jl"
            number = example_source[8:10]
            # generate default main run output file 
            Plotter.close("all")
            include(example_jl_dir * "/" * example_source)
            try
                eval(Meta.parse("$base.main(; Plotter = $Plotter)"))
                figs = Plotter.get_fignums()
                for fig in figs
                    Plotter.figure(fig)
                    imgfile =  example_img_dir * "/" * base * "_$fig.png"
                    Plotter.savefig(imgfile)
                    @info "saved image for example $number to $imgfile"
                end
            catch
                @info "could not generate image for example $number"
            end
        end
    end
end

make_all(; Plotter = PyPlot)