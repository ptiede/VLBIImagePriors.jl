using Reactant, ComradeBase, VLBIImagePriors, Distributions
using Test

@testset "Reactant Ext" begin
    @test Reactant.make_tracer(nothing, ComradeBase.Serial(), nothing, Reactant.ConcreteToTraced; runtime=nothing) isa ComradeBase.ReactantEx

    @testset "GMRF" begin 
        d = GMRF(10.0, (6, 6))
        x = rand(d)
        xr = Reactant.to_rarray(x)

        @jit(logpdf(d, xr)) ≈ logpdf(d, x)
    end


end