
# How to run the examples

Note that all examples are written inside their own modules. To start an example you usually have to include the file and then run the main() function from the example module. The default configuration of the introductory examples is started automatically after include. Some examples have optional argument, e.g. Plotter = PyPlot, to produce some plots with PyPlot (using PyPlot reuqired beforehand) or change other parameters.


Here is an example how to start the example in "examples/doc\_2d\_compressiblestokes.jl":
```
# include example file, e.g.:
include("examples_advanced/doc_2d_compressiblestokes.jl")
# as a result some module is loaded, in this case Example_2DCompressibleStokes

# run without graphics output
Example_2DCompressibleStokes.main()

# run with PyPlot graphics output
using PyPlot
Example_2DCompressibleStokes.main(; Plotter = PyPlot)
```


Also note, that if you make changes to the example files you have to include them again!