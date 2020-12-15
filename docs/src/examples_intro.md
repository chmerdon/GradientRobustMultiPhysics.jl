
# How to run the examples

Note that all examples are written inside their own modules. To start an example you usually have to include the file and then run the main() function from the example module. Some examples have optional argument, e.g. Plotter = PyPlot, produces some plots with PyPlot (using PyPlot reuqired beforehand). Other Plotters are not yet supported.


Here is an example how to start the example in "examples/doc\_2d\_compressiblestokes.jl":
```
# include example file, e.g.:
include("examples/doc_2d_compressiblestokes.jl")
# as a result some module is loaded, in this case Example_2DCompressibleStokes

# run without graphics output
Example_2DCompressibleStokes.main()

# run with PyPlot graphics output
using PyPlot
Example_2DCompressibleStokes.main(; Plotter = PyPlot)
```


Also note, that if you make changes to the example files you have to include them again!