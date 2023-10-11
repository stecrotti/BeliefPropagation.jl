# takes O(|E|^2), could be probably brought down to O(|E|)
function rand_factor_graph(rng::AbstractRNG, nvar::Integer, nfact::Integer, ned::Integer)
    nvar > 0 || throw(ArgumentError("Number of variable nodes must be positive, got $nvar"))
    nfact > 0 || throw(ArgumentError("Number of factor nodes must be positive, got $nfact"))
    ned > 0 || throw(ArgumentError("Number of edges must be positive, got $ned"))
    nedmax = nvar * nfact
    ned ≤ nedmax || throw(ArgumentError("Maximum number of edges is $nvar*$nfact=$nedmax, got $ned"))

    I = zeros(Int, ned)
    J = zeros(Int, ned)
    K = ones(Int, ned)
    n = 1
    while n ≤ ned
        I[n] = rand(rng, 1:nfact)
        J[n] = rand(rng, 1:nvar)
        if !any((i,j) == (I[n], J[n]) for (i,j) in Iterators.take(zip(I,J), n-1))
            n += 1
        end
    end
    A = sparse(I, J, K, nfact, nvar)
    return FactorGraph(A)
end
function rand_factor_graph(nvar::Integer, nfact::Integer, ned::Integer)
    rand_factor_graph(GLOBAL_RNG, nvar, nfact, ned)
end

function rand_factor_graph(rng::AbstractRNG, nvar::Integer, nfact::Integer, p::Real)
    nvar > 0 || throw(ArgumentError("Number of variable nodes must be positive, got $nvar"))
    nfact > 0 || throw(ArgumentError("Number of factor nodes must be positive, got $nfact"))
    0 ≤ p ≤ 1 || throw(ArgumentError("Probability must be in [0,1], got $ned"))

    I = zeros(Int, 0)
    J = zeros(Int, 0)
    for (a, i) in Iterators.product(1:nfact, 1:nvar)
        if rand(rng) < p
            push!(I, a)
            push!(J, i)
        end
    end
    K = ones(Int, length(I))
    A = sparse(I, J, K, nfact, nvar)
    return FactorGraph(A)
end
function rand_factor_graph(nvar::Integer, nfact::Integer, p::Real)
    rand_factor_graph(GLOBAL_RNG, nvar, nfact, p)
end