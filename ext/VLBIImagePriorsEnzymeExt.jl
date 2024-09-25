module VLBIImagePriorsEnzymeExt

using Enzyme
using VLBIImagePriors
using ChainRulesCore 
import TransformVariables as TV
using AbstractFFTs

Enzyme.@import_rrule(typeof(*), AbstractFFTs.Plan, AbstractArray)

end