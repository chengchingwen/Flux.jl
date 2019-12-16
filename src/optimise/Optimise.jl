module Optimise

export train!,
	SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,
  RADAM, Lookahead, InvDecay, ExpDecay, WeightDecay,
  stop, Optimiser

include("optimisers.jl")
include("lookahead.jl")
include("train.jl")

end
