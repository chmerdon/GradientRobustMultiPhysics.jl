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

function make_all(; add_examples::Bool = true)

    #
    # Generate Markdown pages from examples
    #
    if (add_examples)
        example_jl_dir = joinpath(@__DIR__,"..","examples")
        example_jl_dir2 = joinpath(@__DIR__,"..","examples_advanced")
        example_md_dir  = joinpath(@__DIR__,"src","examples")
        example_md_dir2  = joinpath(@__DIR__,"src","examples_advanced")

        for example_source in readdir(example_jl_dir)
            base,ext=splitext(example_source)
            if example_source[1:4] == "doc_" && ext==".jl"
                source_url="https://github.com/chmerdon/GradientRobustMultiPhysics.jl/raw/master/examples/"*example_source
                preprocess(buffer)=replace_source_url(buffer,source_url)|>hashify_block_comments
                try
                    Literate.markdown(joinpath(@__DIR__,"..","examples",example_source),
                                example_md_dir,
                                documenter=false,
                                execute=true,
                                info=false,
                                preprocess=preprocess)
                catch
                    Literate.markdown(joinpath(@__DIR__,"..","examples",example_source),
                                example_md_dir,
                                documenter=false,
                                execute=false,
                                info=false,
                                preprocess=preprocess)
                end
            end
        end

        for example_source in readdir(example_jl_dir2)
            base,ext=splitext(example_source)
            if example_source[1:4] == "doc_" && ext==".jl"
                source_url="https://github.com/chmerdon/GradientRobustMultiPhysics.jl/raw/master/examples_advanced/"*example_source
                preprocess(buffer)=replace_source_url(buffer,source_url)|>hashify_block_comments
                Literate.markdown(joinpath(@__DIR__,"..","examples_advanced",example_source),
                                example_md_dir2,
                                documenter=false,
                                info=false,
                                preprocess=preprocess)
            end
        end
        generated_examples=joinpath.("examples",readdir(example_md_dir))
        generated_examples_advanced=joinpath.("examples_advanced",readdir(example_md_dir2))
    end


    

    makedocs(
        modules=[GradientRobustMultiPhysics],
        format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
        sitename="GradientRobustMultiPhysics.jl",
        authors="Christian Merdon",
        pages = [
            "Home" => "index.md",
            "Problem Description" => Any[
                    "pdedescription.md",
                    "pdeoperators.md",
                    "functionoperators.md",
                    "userdata.md",
                    "boundarydata.md",
                    "globalconstraints.md",
                    "pdeprototypes.md",
            ],
            "Discretisation" => Any[
                    "meshing.md",
                    "fems.md",
                    "fespace.md",
                    "interpolations.md",
                    "quadrature.md",
                    "assembly_details.md"
            ],
            "Solving" => Any[
                    "pdesolvers.md",
                    "timecontrolsolver.md"
                ],
            "Postprocessing" => Any[
                    "viewers.md",
                    "export.md"
                ],
            "Examples" => add_examples ? Any[
                    "examples_intro.md",
                    "Examples (Intro)" => generated_examples,
                    "Examples (Advanced)" => generated_examples_advanced
                ] : "examples_intro.md",
        ]
    )

  #  rm(example_md_dir,recursive=true)
  #  rm(example_md_dir2,recursive=true)
    
end

make_all(; add_examples = true)

deploydocs(
    repo = "github.com/chmerdon/GradientRobustMultiPhysics.jl",
)