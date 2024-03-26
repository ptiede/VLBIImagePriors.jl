function moment_test(d, nsamples=200_000, atol=5e-2)
    # c = cov(d)
    s = reduce(hcat, reshape.(rand(d, nsamples), :))
    # cs = cov(s; dims=2)
    ms = reshape(mean(s; dims=2), size(d))
    # @test isapprox(c, cs; atol)
    @test isapprox(mean(d), ms; atol)
end

function test_interface(d::VLBIImagePriors.MarkovRandomField)
    show(d)
    @inferred VLBIImagePriors.lognorm(d)
    @inferred VLBIImagePriors.unnormed_logpdf(d, rand(d))
    @inferred graph(d)
    @inferred size(d)
    @inferred scalematrix(d)
    c = ConditionalMarkov(typeof(d), Float64, size(d))
    show(c)
    asflat(d)
    @inferred logdet(d)
end

@testset "GMRF" begin

    @testset "MarkovRandomFieldGraph" begin
        g1 = MarkovRandomFieldGraph((4,5); order=1)
        g2 = MarkovRandomFieldGraph(Float64, (4,5); order=1)
        g3 = MarkovRandomFieldGraph(zeros(4, 5); order=1)
        @test g1.G ≈ g2.G
        @test g1.D ≈ g2.D
        @test g1.λQ ≈ g2.λQ
        @test g1.G ≈ g3.G
        @test g1.D ≈ g3.D
        @test g1.λQ ≈ g3.λQ


        g1 = MarkovRandomFieldGraph((4,5); order=2)
        g2 = MarkovRandomFieldGraph(Float64, (4,5); order=2)
        g3 = MarkovRandomFieldGraph(zeros(4, 5); order=2)
        @test g1.G ≈ g2.G
        @test g1.D ≈ g2.D
        @test g1.λQ ≈ g2.λQ
        @test g1.G ≈ g3.G
        @test g1.D ≈ g3.D
        @test g1.λQ ≈ g3.λQ


    end

    test_interface(GaussMarkovRandomField(3.0, rand(10, 8)))
    @testset "Tall" begin
        @testset "Order 1" begin
            mimg = rand(10, 8)
            d1 = GaussMarkovRandomField(3.0, mimg)
            c = MarkovRandomFieldGraph(mimg)
            d2 = GaussMarkovRandomField(3.0, c)

            moment_test(d1)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = scalematrix(d1)
            dd = MvNormalCanon(Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))

            test_rrule(VLBIImagePriors.sq_manoblis, c ⊢ NoTangent(), x, d1.ρ)
        end

        @testset "Order 2" begin
            mimg = rand(10, 8)
            d1 = GaussMarkovRandomField(3.0, mimg; order=2)
            c = MarkovRandomFieldGraph(mimg; order=2)
            d2 = GaussMarkovRandomField(3.0, c)

            moment_test(d1)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = scalematrix(d1)
            dd = MvNormalCanon(Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))

            test_rrule(VLBIImagePriors.sq_manoblis, c ⊢ NoTangent(), x, d1.ρ)

        end

    end

    @testset "Wide" begin
        @testset "Order 1" begin
            mimg = rand(8, 10)
            d1 = GaussMarkovRandomField(3.0, mimg)
            c = MarkovRandomFieldGraph(mimg)
            d2 = GaussMarkovRandomField(3.0, c)

            moment_test(d1)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = scalematrix(d1)
            dd = MvNormalCanon(Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))

            test_rrule(VLBIImagePriors.sq_manoblis, c ⊢ NoTangent(), x, d1.ρ)

        end

        @testset "Order 2" begin
            mimg = rand(8, 10)
            d1 = GaussMarkovRandomField(3.0, mimg; order=2)
            c = MarkovRandomFieldGraph(mimg; order=2)
            d2 = GaussMarkovRandomField(3.0, c)

            moment_test(d1)

            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = scalematrix(d1)
            dd = MvNormalCanon(Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))

            test_rrule(VLBIImagePriors.sq_manoblis, c ⊢ NoTangent(), x, d1.ρ)

        end

    end

    @testset "Equal" begin
        mimg = rand(10, 10)
        d1 = GaussMarkovRandomField(3.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = GaussMarkovRandomField(3.0, c)
        trf, d = matern(size(d2))

        serialize("test.jls" ,trf)
        trf_2 = deserialize("test.jls")
        rm("test.jls")

        x = rand(d)
        @test trf(x, 1.0, 0.1) == trf_2(x, 1.0, 0.1)

        p = trf(rand(d), 1.0, 0.1)
        trf(rand(d), 1.0, 0.1)
        dimg = mean(map(_->trf(rand(d), 1.0, 1.0), 1:1_000_000))
        isapprox(dimg, mimg, atol=1e-2)
        @test size(p) == size(rand(d2))
        logdensityof(d, rand(d))

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        Q = scalematrix(d1)
        dd = MvNormalCanon(Array(Q))

        @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

        @test cov(d1) ≈ cov(dd)
        @test mean(d1) ≈ reshape(mean(dd), size(mimg))

        test_rrule(VLBIImagePriors.sq_manoblis, c ⊢ NoTangent(), x, d1.ρ)

    end


    # @testset "rrules" begin
    #     test_rrule(VLBIImagePriors.igrmf_1n, rand(64,64))
    # end
end


@testset "ExpMRF" begin
    test_interface(ExpMarkovRandomField(3.0, rand(10, 8)))
    @testset "Tall" begin
        mimg = rand(10, 8)
        d1 = ExpMarkovRandomField(3.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = ExpMarkovRandomField(3.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        Q = scalematrix(d1)


    end

    @testset "Wide" begin
        mimg = rand(8, 10)
        d1 = ExpMarkovRandomField(3.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = ExpMarkovRandomField(3.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        Q = scalematrix(d1)
    end

    @testset "Equal" begin
        mimg = rand(10, 10)
        d1 = ExpMarkovRandomField(3.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = ExpMarkovRandomField(3.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
    end

end





@testset "TDistMRF" begin
    test_interface(TDistMarkovRandomField(3.0, 1.0, rand(10, 8)))
    @testset "Tall" begin
        mimg = rand(10, 8)
        d1 = TDistMarkovRandomField(3.0, 1.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = TDistMarkovRandomField(3.0, 1.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        Q = scalematrix(d1)
    end

    @testset "Wide" begin
        mimg = rand(8, 10)
        d1 = TDistMarkovRandomField(3.0, 5.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = TDistMarkovRandomField(3.0, 5.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        Q = scalematrix(d1)
    end

    @testset "Equal" begin
        mimg = rand(10, 10)
        d1 = TDistMarkovRandomField(3.0, 100.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = TDistMarkovRandomField(3.0, 100.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
    end

end

@testset "ConditionalMarkov" begin
    grid = imagepixels(10.0, 5.0, 64, 62)
    c = ConditionalMarkov(GMRF, grid)
    d = c(5.0)
    s = rand(d)

    dm = GaussMarkovRandomField(5.0, (64, 62))
    @test logdensityof(d, s) == logdensityof(dm, s)

    c = ConditionalMarkov(TMRF, grid)
    d = c(5.0, 1.0)
    s = rand(d)

    dm = TDistMarkovRandomField(5.0, 1.0, (64, 62))
    @test logdensityof(d, s) == logdensityof(dm, s)

    c = ConditionalMarkov(EMRF, grid)
    d = c(5.0)
    s = rand(d)

    dm = ExpMarkovRandomField(5.0, (64, 62))
    @test logdensityof(d, s) == logdensityof(dm, s)
end

@testset "Hierarchical Prior" begin
    f(x) = Normal(x[1], exp(x[2]))
    dhyper = product_distribution([Normal(0.0, 1.0), Normal(-1.0, 1.0)])
    dHp = HierarchicalPrior(f, dhyper)
    x0 = rand(dHp)
    fd = f(x0.hyperparams)
    @test logpdf(dHp, x0) ≈ logpdf(fd, x0.params) + logpdf(dhyper, x0.hyperparams)

    x0s = rand(dHp, 1_00)
    x0s = rand(dHp, (10,10))
    asflat(dHp)
end
