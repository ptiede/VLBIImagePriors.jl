using VLBIImagePriors
using Documenter

DocMeta.setdocmeta!(VLBIImagePriors, :DocTestSetup, :(using VLBIImagePriors); recursive=true)

makedocs(;
    modules=[VLBIImagePriors],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/VLBIImagePriors.jl/blob/{commit}{path}#{line}",
    sitename="VLBIImagePriors.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/VLBIImagePriors.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/VLBIImagePriors.jl",
    devbranch="main",
)
