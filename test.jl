using Reactant 
using BenchmarkTools

function testmapreduce(a, A)
    return sum(A) do Ax
        log(a + Ax)
    end
end

function testmap_reduce(a, A)
    return sum(log.(a .+ A))
end

ρr = ConcreteRNumber(2.0)
x = rand(16, 16)

f1 = @compile sync=true testmapreduce(ρr, x)
f2 = @compile sync=true testmap_reduce(ρr, x)



@benchmark f1($ρr, $x)

@benchmark f2($ρr, $x)


A = rand(64, 64)
B = rand(64, 64)

mapreduce(*, +, A, B)


Ar = Reactant.to_rarray(A)
Br = Reactant.to_rarray(B)

mr(A, B) = mapreduce(splat(*), +, zip(A, B))

f = @compile mr(Ar, Br)
