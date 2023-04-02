struct Transport{F,B}
    f::F
    base::B
end

function LogTransport(base)
    return Transport(exp, base)
end

function LogitTransport(base)
    return Transport(alr, base)
end

struct TransportTransform{f} <: TV.VectorTransform
    f::F
    dims::Dims{2}
end
