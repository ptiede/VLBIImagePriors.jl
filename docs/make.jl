using RadioImagePriors
using Documenter

DocMeta.setdocmeta!(RadioImagePriors, :DocTestSetup, :(using RadioImagePriors); recursive=true)

makedocs(;
    modules=[RadioImagePriors],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/RadioImagePriors.jl/blob/{commit}{path}#{line}",
    sitename="RadioImagePriors.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/RadioImagePriors.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/RadioImagePriors.jl",
    devbranch="main",
)
