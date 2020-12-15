using Documenter
using Literate
using ExtendableSparse
using ExtendableGrids

using GradientRobustMultiPhysics


# Turn block comments starting in the first column into "normal" hash comments
# as block comments currently are not handled by Literate.jl.
function hashify_block_comments(input)
    lines_in = collect(eachline(IOBuffer(input)))
    lines_out=IOBuffer()
    line_number=0
    in_block_comment_region=false
    for line in lines_in
        line_number+=1
        if occursin(r"^#=", line)
            if in_block_comment_region
                error("line $(line_number): already in block comment region\n$(line)")
            end
            println(lines_out,replace(line,r"^#=" => "#"))
            in_block_comment_region=true
        elseif occursin(r"^=#", line)
            if !in_block_comment_region
                error("line $(line_number): not in block comment region\n$(line)")
            end
            println(lines_out,replace(line,r"^=#" => "#"))
            in_block_comment_region=false
        else
            if in_block_comment_region
                println(lines_out,"# "*line)
            else
                println(lines_out,line)
            end
        end
    end
    return String(take!(lines_out))
end

#
# Replace SOURCE_URL marker with github url of source
#
function replace_source_url(input,source_url)
    lines_in = collect(eachline(IOBuffer(input)))
    lines_out=IOBuffer()
    for line in lines_in
        println(lines_out,replace(line,"SOURCE_URL" => source_url))
    end
    return String(take!(lines_out))
end

function make_all()

    #
    # Generate Markdown pages from examples
    #
    example_jl_dir = joinpath(@__DIR__,"..","examples")
    example_md_dir  = joinpath(@__DIR__,"src","examples")
    
    for example_source in readdir(example_jl_dir)
        base,ext=splitext(example_source)
        if example_source[1:4] == "doc_" && ext==".jl"
            source_url="https://github.com/chmerdon/GradientRobustMultiPhysics.jl/raw/master/examples/"*example_source
            preprocess(buffer)=replace_source_url(buffer,source_url)|>hashify_block_comments
            Literate.markdown(joinpath(@__DIR__,"..","examples",example_source),
                              example_md_dir,
                              documenter=false,
                              info=false,
                              preprocess=preprocess)
        end
    end
    generated_examples=joinpath.("examples",readdir(example_md_dir))
    pushfirst!(generated_examples, "examples_intro.md")

    makedocs(
        modules=[GradientRobustMultiPhysics],
        format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
        sitename="GradientRobustMultiPhysics.jl",
        authors="Christian Merdon",
        pages = [
            "Home" => "index.md",
            "Implemented Finite Elements" => "fems.md",
            "FE Spaces and Arrays" => "fespace.md",
            "FE Interpolation" => "interpolations.md",
            "Function Operators" => "functionoperators.md",
            "PDE Description" => "pdedescription.md",
            "PDE Prototypes" => "pdeprototypes.md",
            "PDE Solvers" => "pdesolvers.md",
            "User Data" => "userdata.md",
            "Assembly Details" => "assembly_details.md",
            "Meshing" => "meshing.md",
            "Quadrature" => "quadrature.md",
            "Export/Viewers" => "viewers.md",
            "Examples" => generated_examples
        ]
    )

    rm(example_md_dir,recursive=true)
    
end

make_all()

deploydocs(
    repo = "github.com/chmerdon/GradientRobustMultiPhysics.jl",
)