function moment_test(d, nsamples = 200_000, atol = 5.0e-2)
    # c = cov(d)
    s = reduce(hcat, reshape.(rand(d, nsamples), :))
    # cs = cov(s; dims=2)
    ms = reshape(mean(s; dims = 2), size(d))
    # @test isapprox(c, cs; atol)
    return @test isapprox(mean(d), ms; atol)
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
    return @inferred logdet(d)
end

@testset "GMRF" begin

    @testset "MarkovRandomFieldGraph" begin
        g1 = MarkovRandomFieldGraph((4, 5); order = 1)
        g2 = MarkovRandomFieldGraph(Float64, (4, 5); order = 1)
        g3 = MarkovRandomFieldGraph(zeros(4, 5); order = 1)
        @test g1.G ≈ g2.G
        @test g1.D ≈ g2.D
        @test g1.λQ ≈ g2.λQ
        @test g1.G ≈ g3.G
        @test g1.D ≈ g3.D
        @test g1.λQ ≈ g3.λQ


        g1 = MarkovRandomFieldGraph((4, 5); order = 2)
        g2 = MarkovRandomFieldGraph(Float64, (4, 5); order = 2)
        g3 = MarkovRandomFieldGraph(zeros(4, 5); order = 2)
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
            d1 = GaussMarkovRandomField(3.0, mimg; order = 2)
            c = MarkovRandomFieldGraph(mimg; order = 2)
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
            d1 = GaussMarkovRandomField(3.0, mimg; order = 2)
            c = MarkovRandomFieldGraph(mimg; order = 2)
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


        @testset "Order 3" begin
            mimg = rand(8, 10)
            d1 = GaussMarkovRandomField(3.0, mimg; order = 3)
            c = MarkovRandomFieldGraph(mimg; order = 3)
            d2 = GaussMarkovRandomField(3.0, c)


            x = rand(d1)
            @test logpdf(d1, x) ≈ logpdf(d2, x)
            Q = scalematrix(d1)
            dd = MvNormalCanon(Array(Q))

            @test logpdf(d1, x) ≈ logpdf(dd, reshape(x, :))

            @test cov(d1) ≈ cov(dd)
            @test mean(d1) ≈ reshape(mean(dd), size(mimg))
        end


    end

    @testset "Equal" begin
        mimg = rand(10, 10)
        d1 = GaussMarkovRandomField(3.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = GaussMarkovRandomField(3.0, c)
        trf, d = matern(size(d2))

        cd = ascube(d)
        x = rand(dimension(cd))
        @test inverse(cd, transform(cd, x)) ≈ x
        x100 = rand(dimension(cd), 10000)
        p100 = transform.(Ref(cd), eachcol(x100))
        ms = mean(p100)
        ss = std(p100)
        @test isapprox(ms, zeros(100), atol = 10 / sqrt(1000))
        @test isapprox(ss, ones(100), atol = 50 / sqrt(1000))


        serialize("test.jls", trf)
        trf_2 = deserialize("test.jls")
        rm("test.jls")

        x = rand(d)
        @test trf(x, 1.0, 0.1) == trf_2(x, 1.0, 0.1)
        @test trf(x, (5.0, 5.0), 0.0, 1.0) == trf(x, 5.0, 1.0)

        p = trf(rand(d), 1.0, 0.1)
        trf(rand(d), 1.0, 0.1)
        dimg = mean(map(_ -> trf(rand(d), 1.0, 1.0), 1:1_000_000))
        isapprox(dimg, mimg, atol = 1.0e-2)
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
        d1 = TDistMarkovRandomField(3.0, 1.0, mimg)
        c = MarkovRandomFieldGraph(mimg)
        d2 = TDistMarkovRandomField(3.0, 2.0, c)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
    end

    @testset "TDist order" begin
        mimg = rand(10, 10)
        d1 = TDistMarkovRandomField(3.0, 1.0, mimg; order = 1)
        @test all(==(Inf), mean(d1))
        @test all(==(Inf), cov(d1))
        @test all(x -> x === NaN, invcov(d1))

        d2 = TDistMarkovRandomField(3.0, 2.0, mimg; order = 1)
        @test all(==(0), mean(d2))
        @test all(==(Inf), cov(d2))
        @test all(x -> x === NaN, invcov(d2))


        d3 = TDistMarkovRandomField(3.0, 3.0, mimg; order = 1)
        @test all(==(0), mean(d3))
        @test all(x -> x != (Inf), cov(d3))
        @test all(x -> !(x === NaN), invcov(d3))

    end

    @testset "CauchyMRF" begin
        img = rand(10, 10)
        d1 = CauchyMarkovRandomField(3.0, img)
        d2 = CauchyMarkovRandomField(3.0, VLBIImagePriors.graph(d1))
        d3 = CauchyMarkovRandomField(3.0, size(img))

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        @test logpdf(d1, x) ≈ logpdf(d3, x)


        d1 = CauchyMarkovRandomField(3.0, img; order = 2)
        d2 = CauchyMarkovRandomField(3.0, VLBIImagePriors.graph(d1))
        d3 = CauchyMarkovRandomField(3.0, size(img); order = 2)

        x = rand(d1)
        @test logpdf(d1, x) ≈ logpdf(d2, x)
        @test logpdf(d1, x) ≈ logpdf(d3, x)
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
    @test length(dHp) == 3
    x0 = rand(dHp)
    fd = f(x0.hyperparams)
    @test logpdf(dHp, x0) ≈ logpdf(fd, x0.params) + logpdf(dhyper, x0.hyperparams)

    x0s = rand(dHp, 1_00)
    x0s = rand(dHp, (10, 10))
    asflat(dHp)
    show(dHp)
end


@testset "Noncenter Markov Gaussian" begin

    function testnoncenter(d)
        ρ = d.ρ
        t, ds = standardize(d)

        serialize("test.jls", t)
        t2 = deserialize("test.jls")


        @test size(t) == size(d)
        @test size(ds) == size(d)
        @test d.ρ == ρ

        s = rand(d, 10000)
        ss = centerdist.(Ref(t), ρ, rand(ds, 100000))


        @test all(x -> isapprox(x[1], x[2]; atol = 1.0e-1), zip(mean(s), mean(ss)))
        @test all(x -> isapprox(x[1], x[2]; atol = 1.0e-1), zip(std(s), std(ss)))

        u = noncenterdist.(Ref(t), ρ, rand(d, 5000))
        @test all(x -> isapprox(x, 0; atol = 1.0e-1), mean(u))
        @test all(x -> isapprox(x, 1; atol = 1.0e-1), std(u))
    end

    ρ = 8.0
    d = GMRF(ρ, (20, 20); order = 1)
    testnoncenter(d)

    d = GMRF(ρ, (20, 20); order = 2)
    testnoncenter(d)

    d = GMRF(ρ, (20, 20); order = 3)
    testnoncenter(d)
end
