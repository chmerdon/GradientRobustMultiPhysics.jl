using PyPlot
using GLMakie
#using Plots
using ExtendableSparse
using ExtendableGrids
using GradientRobustMultiPhysics

function make_all(which = []; Plotter2D = PyPlot, Plotter3D = GLMakie)

    # generate images for all examples
    # (this is called only manually at the moment)
    example_jl_dir = joinpath(@__DIR__,"..","examples")
    example_img_dir  = joinpath(@__DIR__,"src","images")

    for example_source in readdir(example_jl_dir)
        base,ext=splitext(example_source)
        if example_source[1:7] == "Example" && ext==".jl"
            number = example_source[8:10]
            if example_source[8] == 'A'
                dim = 2
            else
                dim = parse(Int,example_source[8])
            end
            if dim == 3
                Plotter = Plotter3D
            else
                Plotter = Plotter2D
            end
            if number in which || which == []
                # generate default main run output file 
                Plotter.close("all")
                include(example_jl_dir * "/" * example_source)
                #try
                    eval(Meta.parse("$base.main(; Plotter = $Plotter)"))
                    if Plotter == Plotter2D
                        figs = Plotter.get_fignums()
                        for fig in figs
                            imgfile =  example_img_dir * "/" * base * "_$fig.png"
                            Plotter.figure(fig)
                            Plotter.savefig(imgfile)
                            @info "saved image for example $number to $imgfile"
                        end
                    elseif Plotter == Plotter3D
                        imgfile =  example_img_dir * "/" * base * "_1.png"
                        scene = Makie.current_scene()
                        Plotter.save(imgfile,scene)
                        @info "saved image for example $number to $imgfile"
                    end
                #catch
                #    @info "could not generate image for example $number"
                #end
            end
        end
    end
end