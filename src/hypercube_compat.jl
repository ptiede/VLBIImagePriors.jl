# Backwards-compatible HypercubeTransform-style entry points.
#
# Historically callers (notably Comrade) built parameter-space transforms with
# `asflat(d)` / `ascube(d)` from HypercubeTransform and applied them with
# `transform` / `inverse` / `transform_and_logjac` (re-exported from
# TransformVariables). That dependency is gone; these names now delegate directly to
# ProbabilityTransports' transport interface so the public API is unchanged. Both
# `asflat`/`ascube` return a `TransportedDistribution`, and the verbs below extend
# the canonical TransformVariables generics on that type so existing Comrade call
# sites keep dispatching to them:
#
#   transform(t, y)            -> transport(t, y)
#   inverse(t, x)              -> pullback(t, x)
#   dimension(t)               -> dimension(t)        (unchanged; from PT)
#
# Note: the old `transform_and_logjac` alias is intentionally not provided. PT no
# longer exposes a bare Jacobian (`transport_and_logjac` was removed); the replacement
# `transport_and_logdensity` returns the pulled-back *prior log-density*, not a Jacobian,
# so silently re-aliasing the old name would change its meaning. Call
# `transport_and_logdensity` directly instead.
#
# Scalar vs array routing is handled inside PT (a 0-dim distribution maps through a
# scalar node), so these wrappers stay distribution-agnostic.

import TransformVariables: transform, inverse, dimension

export asflat, ascube, transform, inverse, dimension

"""
    asflat(d)

Return the transport from the unconstrained real space (`TVFlat`) to the support of
`d`. Backwards-compatible alias for `transport_to(d, TVFlat())`.
"""
asflat(d) = transport_to(d, TVFlat())

"""
    ascube(d)

Return the transport from the unit hypercube (`StdUniform`) to the support of `d`.
Backwards-compatible alias for `transport_to(d, StdUniform())`.
"""
ascube(d) = transport_to(d, StdUniform())


# ----- HypercubeTransform verb aliases ------------------------------------
# Extend the TransformVariables generics on `TransportedDistribution` so callers
# that still say `transform`/`inverse`/`transform_and_logjac` (as HypercubeTransform
# did) dispatch onto the PT transport interface.

"""
    transform(t::TransportedDistribution, y)

Apply the transport `t` to the latent point `y`. Alias for `transport(t, y)`.
"""
transform(t::TransportedDistribution, y) = transport(t, y)

"""
    inverse(t::TransportedDistribution, x)

Map `x` back to a canonical latent point. Alias for `pullback(t, x)`.
"""
inverse(t::TransportedDistribution, x) = pullback(t, x)

"""
    dimension(t::TransportedDistribution)

Number of latent coordinates `t` consumes. Extends `TransformVariables.dimension`
onto `TransportedDistribution` so old call sites (which used the TV generic) keep
working; delegates to `ProbabilityTransports.dimension`.
"""
dimension(t::TransportedDistribution) = ProbabilityTransports.dimension(t)
