var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API reference","title":"API reference","text":"CurrentModule = BeliefPropagation","category":"page"},{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/#Docstrings","page":"API reference","title":"Docstrings","text":"","category":"section"},{"location":"api/#Belief-Propagation","page":"API reference","title":"Belief Propagation","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [BeliefPropagation]","category":"page"},{"location":"api/#BeliefPropagation.BP","page":"API reference","title":"BeliefPropagation.BP","text":"BP{F<:BPFactor, FV<:BPFactor, M, MB, G<:FactorGraph}\n\nA type representing the state of the Belief Propagation algorithm.\n\nFields\n\ng: a FactorGraph(see IndexedFactorGraphs.jl)\nψ: a vector of BPFactor representing the factors {ψₐ(xₐ)}ₐ\nϕ: a vector of BPFactor representing the single-variable factors {ϕᵢ(xᵢ)}ᵢ\nu: messages from factor to variable\nh: messages from variable to factor\nb: beliefs\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.BP-Tuple{IndexedFactorGraphs.AbstractFactorGraph, AbstractVector{<:BPFactor}, Any}","page":"API reference","title":"BeliefPropagation.BP","text":"BP(g::FactorGraph, ψ::AbstractVector{<:BPFactor}, states; ϕ)\n\nConstructor for the BP type.\n\nArguments\n\ng: a FactorGraph\nψ: a vector of BPFactor representing the factors {ψₐ(xₐ)}ₐ\nstates: an iterable of integers of length equal to the number of variable nodes specifyig the number of values each variable can take \nϕ: (optional) a vector of BPFactor representing the single-variable factors {ϕᵢ(xᵢ)}ᵢ\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.BPFactor","page":"API reference","title":"BeliefPropagation.BPFactor","text":"BPFactor\n\nAn abstract type representing a factor.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.BeliefConvergence","page":"API reference","title":"BeliefPropagation.BeliefConvergence","text":"BeliefConvergence\n\nCalled after an iteration in a callback, it computes the maximum absolute change in beliefs with (::BeliefConvergence)(::BP, errv, errf, errb, it)\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Callback","page":"API reference","title":"BeliefPropagation.Callback","text":"abstract type Callback\n\nSubtypes can be used as callbacks during the iterations.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.ConvergenceChecker","page":"API reference","title":"BeliefPropagation.ConvergenceChecker","text":"abstract type ConvergenceChecker\n\nSubtypes such as MessageConvergence compute convergence errors\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Decimation","page":"API reference","title":"BeliefPropagation.Decimation","text":"Decimation(n, maxiter, tol)\n\nReturn an instance of the Decimation callback.\n\nArguments\n\nn: total number of variables\nmaxiter: maximum number of iterations\ntol: tolerance for convergence check\n\nOptional arguments\n\nconv_checker: a ConvergenceChecker\nsoftinf: the real value used to fix variables\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Decimation-2","page":"API reference","title":"BeliefPropagation.Decimation","text":"Decimation <: Callback\n\nA callback that implements the decimation procedure: whenever the desired convergence tolerance has been reached, the variable with the most biased belief is fixed to that value by modifying the corresponding ϕ factor. The procedure is repeated until all variables are fixed. The recommended constructor is Decimation(n::Integer, tol::Real).\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.MessageConvergence","page":"API reference","title":"BeliefPropagation.MessageConvergence","text":"MessageConvergence\n\nCalled after an iteration in a callback, it computes the maximum absolute change in messages with (::MessageConvergence)(::BP, errv, errf, errb, it)\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.ProgressAndConvergence","page":"API reference","title":"BeliefPropagation.ProgressAndConvergence","text":"ProgressAndConvergence(maxiter, tol)\n\nReturn an instance of the ProgressAndConvergence callback.\n\nArguments\n\nmaxiter: maximum number of iterations\ntol: tolerance for convergence check\n\nOptional arguments\n\nconv_checker: a ConvergenceChecker\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.ProgressAndConvergence-2","page":"API reference","title":"BeliefPropagation.ProgressAndConvergence","text":"ProgressAndConvergence <: Callback\n\nA basic callback that prints a progress bar and checks convergence.ù The recommended constructor is ProgressAndConvergence(maxiter::Integer, tol::Real).\n\nFields\n\nprog: a Progress from ProgressMeter.jl\ntol: the tolerance below which BP is considered at a fixed point\nconv_checker: a ConvergenceChecker \n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.TabulatedBPFactor","page":"API reference","title":"BeliefPropagation.TabulatedBPFactor","text":"TabulatedBPFactor\n\nA type of BPFactor constructed by specifying the output to any input in a tabular fashion via an array values.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.UniformFactor","page":"API reference","title":"BeliefPropagation.UniformFactor","text":"UniformFactor\n\nA type of BPFactor which returns the same value for any input: it behaves as if it wasn't even there. It is used as the default for single-variable factors\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.avg_energy-Tuple{Function, BP}","page":"API reference","title":"BeliefPropagation.avg_energy","text":"avg_energy([f], bp::BP)\n\nReturn the average energy\n\nlangle E rangle =sum_asum_underlinex_ab_a(underlinex_a) left-logpsi_a(underlinex_a)right + sum_isum_x_ib_i(x_i) left-logphi_i(x_i)right\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.beliefs-Tuple{Function, BP}","page":"API reference","title":"BeliefPropagation.beliefs","text":"beliefs([f], bp::BP)\n\nReturn single-variable beliefs b_i(x_i)_i.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.bethe_free_energy-Tuple{Function, BP}","page":"API reference","title":"BeliefPropagation.bethe_free_energy","text":"bethe_free_energy([f], bp::BP)\n\nReturn the bethe free energy\n\nF=sum_asum_underlinex_ab_a(underlinex_a) left-logfracb_a(underlinex_a)psi_a(underlinex_a)right + sum_isum_x_ib_i(x_i) left-logfracb_i(x_i)^1-lvertpartial irvertphi_i(x_i)right\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.energy-Tuple{BP, Any}","page":"API reference","title":"BeliefPropagation.energy","text":"energy(bp::BP, x)\n\nReturn the energy\n\nE(underlinex)=sum_a left-logpsi_a(underlinex_a)right + sum_i left-logphi_i(x_i)right\n\nof configuration x.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.evaluate-Tuple{BP, Any}","page":"API reference","title":"BeliefPropagation.evaluate","text":"evaluate(bp::BP, x)\n\nReturn the unnormalized probability prod_apsi_a(underlinex_a)prod_iphi_i(x_i) of configuration x.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.factor_beliefs-Tuple{Function, BP}","page":"API reference","title":"BeliefPropagation.factor_beliefs","text":"factor_beliefs([f], bp::BP)\n\nReturn factor beliefs b_a(underlinex_a)_a.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.iterate!-Tuple{BP}","page":"API reference","title":"BeliefPropagation.iterate!","text":"iterate!(bp::BP; kwargs...)\n\nRun BP.\n\nOptional arguments\n\nupdate_variable!: the function that computes and updates variable-to-factor messages\nupdate_factor!: the function that computes and updates factor-to-variable messages\nmaxiter: maximum number of iterations\ntol: convergence check parameter\ndamp: damping parameter\nrein: reinforcement parameter\ncallbacks: a vector of callbacks. By default a ProgressAndConvergence\nextra arguments to be passed to custom update_variable! and update_factor!\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.iterate_ms!-Tuple{BP}","page":"API reference","title":"BeliefPropagation.iterate_ms!","text":"iterate_ms!(bp::BP; kwargs...)\n\nRuns the max-sum algorithm (BP at zero temperature). \n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.nstates-Tuple{BP, Integer}","page":"API reference","title":"BeliefPropagation.nstates","text":"nstates(bp::BP, i::Integer)\n\nReturn the number of values taken by variable i.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.randomize!-Tuple{Random.AbstractRNG, BP}","page":"API reference","title":"BeliefPropagation.randomize!","text":"randomize!([rng], bp::BP)\n\nFill messages and belief with random values\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.reset!-Tuple{BP}","page":"API reference","title":"BeliefPropagation.reset!","text":"reset!(bp::BP)\n\nReset all messages and beliefs to zero\n\n\n\n\n\n","category":"method"},{"location":"api/#Models","page":"API reference","title":"Models","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [BeliefPropagation.Models]","category":"page"},{"location":"api/#BeliefPropagation.Models.ColoringCoupling","page":"API reference","title":"BeliefPropagation.Models.ColoringCoupling","text":"ColoringCoupling\n\nA type of BPFactor representing a factor in the coloring problem. It always involves two (discrete) incident variables x_i, x_j. The factor evaluates to psi(x_ix_j)=1-delta(x_ix_j)\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Models.IsingCoupling","page":"API reference","title":"BeliefPropagation.Models.IsingCoupling","text":"IsingCoupling\n\nA type of BPFactor representing a factor in an Ising distribution. It involves pm 1 variables boldsymbolsigma_a=sigma_1 sigma_2 ldots. The factor evaluates to psi(boldsymbolsigma_a)=e^beta J prod_iin asigma_i. A particular case is the pairwise interaction where a=ij is a pair of vertices involved in an edge (ij).\n\nFields\n\nβJ: coupling strength.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Models.IsingField","page":"API reference","title":"BeliefPropagation.Models.IsingField","text":"IsingField\n\nA type of BPFactor representing a single-variable external field in an Ising distribution. The factor evaluates to psi(sigma_i)=e^beta h sigma_i.\n\nFields\n\nβh: field strength.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Models.KSATClause","page":"API reference","title":"BeliefPropagation.Models.KSATClause","text":"KSATClause\n\nA type of BPFactor representing a clause in a k-SAT formula. It involves 01 variables boldsymbolx_a=x_1 x_2 ldots x_k. The factor evaluates to psi_a(boldsymbolx_a)=1 - prod_iin adelta(x_i J^a_i).\n\nFields\n\nJ: a vector of booleans.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Models.SoftColoringCoupling","page":"API reference","title":"BeliefPropagation.Models.SoftColoringCoupling","text":"SoftColoringCoupling\n\nA soft version of ColoringCoupling. It always involves two (discrete) incident variables x_i, x_j. The factor evaluates to psi(x_ix_j)=e^-betadelta(x_ix_j)\n\nFields\n\nβ: the real parameter controlling the softness. A ColoringCoupling is recovered in the large β limit.\n\n\n\n\n\n","category":"type"},{"location":"api/#BeliefPropagation.Models.fast_ising_bp","page":"API reference","title":"BeliefPropagation.Models.fast_ising_bp","text":"fast_ising_bp(g::AbstractFactorGraph, ψ::Vector{<:IsingCoupling}, [ϕ])\n\nReturn a BP instance with Ising factors and messages and beliefs in log-ratio format:\n\nbeginalign*\n\tm_ato i(sigma_i) propto e^u_ato isigma_i\n\tu_ato i = frac12logfracm_ato i(+1)m_ato i(+1)\nendalign*\n\n\n\n\n\n","category":"function"},{"location":"api/#BeliefPropagation.Models.fast_ksat_bp","page":"API reference","title":"BeliefPropagation.Models.fast_ksat_bp","text":"fast_ksat_bp(g::AbstractFactorGraph, ψ::Vector{<:KSATClause}, [ϕ])\n\nReturn a specialized BP instance with KSATClause and messages encoded as reals instead of vectors: only the probability of x=1 is stored.  ```\n\n\n\n\n\n","category":"function"},{"location":"api/#Test-utilities","page":"API reference","title":"Test utilities","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [BeliefPropagation.Test]","category":"page"},{"location":"api/#BeliefPropagation.TabulatedBPFactor-Tuple{BPFactor, Any}","page":"API reference","title":"BeliefPropagation.TabulatedBPFactor","text":"TabulatedBPFactor(f::BPFactor, states)\n\nConstruct a TabulatedBPFactor out of any BPFactor. Used mostly for tests.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.exact_avg_energy-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.exact_avg_energy","text":"exact_avg_energy(bp::BP; p_exact = exact_prob(bp))\n\nExhaustively compute the average energy (minus the log of the unnormalized probability weight).\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.exact_factor_marginals-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.exact_factor_marginals","text":"exact_factor_marginals(bp::BP; p_exact = exact_prob(bp))\n\nExhaustively compute marginal distributions for each factor.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.exact_marginals-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.exact_marginals","text":"exact_marginals(bp::BP; p_exact = exact_prob(bp))\n\nExhaustively compute marginal distributions for each variable.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.exact_minimum_energy-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.exact_minimum_energy","text":"exact_minimum_energy(bp::BP)\n\nExhaustively compute the minimum energy (minus the log of the unnormalized probability weight).\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.exact_normalization-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.exact_normalization","text":"exact_normalization(bp::BP)\n\nExhaustively compute the normalization constant.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.exact_prob-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.exact_prob","text":"exact_prob(bp::BP; Z = exact_normalization(bp))\n\nExhaustively compute the probability of each possible configuration of the variables.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.make_generic-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.make_generic","text":"make_generic(bp::BP)\n\nReturn the corresponding BP with plain TabulatedBPFactor as factors. Used to test specific implementations.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.rand_bp-Tuple{Random.AbstractRNG, IndexedFactorGraphs.AbstractFactorGraph, Any}","page":"API reference","title":"BeliefPropagation.Test.rand_bp","text":"rand_bp([rng], g::FactorGraph, states)\n\nReturn a BP with random factors.\n\nstates is an iterable containing the number of values that can be taken by each variable.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.rand_factor-Tuple{Random.AbstractRNG, Any}","page":"API reference","title":"BeliefPropagation.Test.rand_factor","text":"rand_factor([rng,], states)\n\nReturn a random BPFactor whose domain is specified by the iterable states.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.test_observables_bp-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.test_observables_bp","text":"test_observables_bp(bp::BP; kwargs...)\n\nTest beliefs_bp, factor_beliefs_bp and bethe_free_energy_bp against the same quantities computed exactly by exhaustive enumeration.\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.test_observables_bp_generic-Tuple{BP, BP}","page":"API reference","title":"BeliefPropagation.Test.test_observables_bp_generic","text":"test_observables_bp_generic(bp::BP, bp_generic::BP; kwargs...)\n\nTest beliefs_bp, factor_beliefs_bp and bethe_free_energy_bp against the same quantities on a generic version. See also make_generic\n\n\n\n\n\n","category":"method"},{"location":"api/#BeliefPropagation.Test.test_za-Tuple{BP}","page":"API reference","title":"BeliefPropagation.Test.test_za","text":"test_za(bp::BP)\n\nTest a specific implementation of compute_za against the naive one.\n\n\n\n\n\n","category":"method"},{"location":"api/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"This package is made with the goal of allowing users to define their own BPFactors and the corresponding specialized updates. In particular, the bottleneck of BP compuations typically is the update of factor-to-variable messages, whose cost grows exponentially with the degree of the factor node.","category":"page"},{"location":"custom_factors/#Specialized-factor-updates","page":"Custom models","title":"Specialized factor updates","text":"","category":"section"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"For many models one can devise more efficient implementations. To integrate these with the BeliefPropagation.jl API, users can define a new BPFactor and override existing methods by dispatching on the factor type. The minimum required API for a custom MyFactor <: BPFactor is:","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"(f::MyFactor)(x): a functor evaluating the factor for a given input x\nBeliefPropagation.compute_za(bp::{<:MyFactor}, a::Integer, msg_in) which computes the factor normalization","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"beginequation*\nz_a = sum_underlinex_a psi_a(underlinex_a) prod_iinpartial a h_ito a(x_i)\nendequation*","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"where h_ito a are the incoming variable-to-factor messages.","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"And... that's it! From the computation of z_a, the one for outgoing messages is obtained under the hood using automatic differentiation.","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"For example, here is the efficient implementation for factors representing K-SAT formulas:","category":"page"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"struct KSATClause{T}  <: BPFactor where {T<:AbstractVector{<:Bool}}\n    J :: T \nend\n\nfunction (f::KSATClause)(x) \n    isempty(x) && return 1.0\n    return any(xᵢ - 1 != Jₐᵢ for (xᵢ, Jₐᵢ) in zip(x, f.J)) |> float\nend\n\nfunction BeliefPropagation.compute_za(bp::BP{<:KSATClause}, a::Integer, \n        msg_in::AbstractVector{<:AbstractVector{<:Real}})\n    ψₐ = bp.ψ[a]\n    isempty(msg_in) && return one(eltype(ψₐ))\n    z1 = prod(sum(hᵢₐ) for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))\n    z2 = prod(hᵢₐ[Jₐᵢ+1] for (hᵢₐ, Jₐᵢ) in zip(msg_in, ψₐ.J))\n    return z1 - z2\nend","category":"page"},{"location":"custom_factors/#Further-optimizations","page":"Custom models","title":"Further optimizations","text":"","category":"section"},{"location":"custom_factors/","page":"Custom models","title":"Custom models","text":"BeliefPropagation.jl by default represents messages as Vectors of floats. However, for problems with binary variables, messages are binary distributions and only need one parameter to be specified. In these case, one can override the functions that perform the update for a simplified type of messages, as well as a specific type of factor. As an example, see the efficient implementation of BP for the Ising model provided here.","category":"page"},{"location":"#BeliefPropagation","page":"Home","title":"BeliefPropagation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Aqua)","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"⚠️ This package is heavily work in progress, some breaking changes should be expected.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package implements a generic version of the Belief Propagation (BP) algorithm for the approximation of probability distributions factorized on a graph","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginequation\np(x_1x_2ldotsx_n) propto prod_ain F psi_a(underlinex_a) prod_iin V phi_i(x_i) \nendequation","category":"page"},{"location":"","page":"Home","title":"Home","text":"where F is the set of factors, V the set of variables, and underlinex_a is the set of variables involved in factor a.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"import Pkg; Pkg.add(\"BeliefPropagation\")","category":"page"},{"location":"#Quickstart","page":"Home","title":"Quickstart","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Check out the examples folder.","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The goal of this package is to provide a simple, flexible, and ready-to-use interface to the BP algorithm. It is enough for the user to provide the factor graph (encoded in an adjacency matrix or as a Graphs.jl graph) and the factors, everything else is taken care of.","category":"page"},{"location":"","page":"Home","title":"Home","text":"At the same time, the idea is that refinements can be made to improve performance on a case-by-case basis. For example, messages are stored as Vectors by default, but when working with binary variables, one real number is enough, allowing for considerable speed-ups (see the Ising example). Also, a version of BP for continuous variables such as Gaussian BP can be introduced in the framework, although it is not yet implemented.","category":"page"},{"location":"#See-also","page":"Home","title":"See also","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"BeliefPropagation.jl: implements BP for the Ising model and the matching problem.\nFactorGraph.jl: implements Gaussian BP and other message-passing algorithms.\nITensorNetworks.jl: implements BP as a technique for approximate tensor network contraction.\nReactiveMP.jl: allows to solve Bayesian inference problems using message-passing.\nCodingTheory: has a specialized implementation of BP for the coding problem","category":"page"}]
}
