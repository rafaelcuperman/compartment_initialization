using DrWatson
@quickactivate "compartment-initialization"

using Turing
using DeepCompartmentModels
using Plots

function plot_priors_u0(priors)
    u01_prior = priors["u01_prior"]
    x = range(u01_prior.lower - 20, stop=u01_prior.upper + 20, length=1000);
    plt_u01 = plot(x, pdf.(u01_prior, x), title="u01 prior", label="", yticks=nothing);

    u02_prior = priors["u02_prior"]
    x = range(u02_prior.lower - 20, stop=u02_prior.upper + 20, length=1000);
    plt_u02 = plot(x, pdf.(u02_prior, x), title="u02 prior", label="", yticks=nothing);

    etas_prior = priors["etas_prior"]
    x = range(-3, stop=3, length=1000);
    y = range(-3, stop=3, length=1000);
    X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y];
    Z = [pdf(etas_prior, [X[i, j], Y[i, j]]) for i in 1:size(X, 1), j in 1:size(X, 2)];
    plt_etas = contour(x, y, Z, xlabel="eta[1]", ylabel="eta[2]", title="Etas prior", label="", colorbar=nothing);

    return plot(plt_u01, plt_u02, plt_etas, layout=(3,1), size = (800, 600))
end

function run_chain(pkmodel::Function, ind::BasicIndividual, I::AbstractMatrix, priors::Dict, args...; algo=NUTS(0.65), iters::Int=2000, chains::Int=3, sigma=5, kwargs...)
    @model function model_u0(pkmodel, ind, I, priors, args...; kwargs...)
        u01 ~ priors["u01_prior"]
        u02 ~ priors["u02_prior"]
        etas ~ priors["etas_prior"] 
    
        u0_ = [u01, u02]
    
        predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=etas, u0=u0_, tspan=(-0.1, ind.t[end] + 10), kwargs...)
    
        ind.y ~ MultivariateNormal(vec(predicted), sigma)
    
        return nothing
    end

    # Build model
    model = model_u0(pkmodel, ind, I, priors);

    # Sample from model
    chain = sample(model, algo, MCMCSerial(), iters, chains; progress=true);
    return chain
end

# Sample n items from the posterior and simulate the curves
function sample_posterior(chain, ind::BasicIndividual, I::AbstractMatrix; n::Int=100, saveat=ind.t)
    # Sample n u0s
    posterior_samples = sample(chain[[:u01, :u02, Symbol("etas[1]"), Symbol("etas[2]")]], n, replace=false);

    saveat = saveat isa AbstractVector ? saveat : collect(0:saveat:ind.t[end] + 10);

    list_predicted = []
    ps = []
    # Plot solutions for all the sampled parameters
    plt = plot(title="n =  $n")
    for p in eachrow(Array(posterior_samples))
        sample_u01, sample_u02, sample_eta1, sample_eta2 = p
        push!(ps, p)

        # Set initial values
        u0_ = [sample_u01, sample_u02]

        predicted = pkmodel(ind, I, saveat; save_idxs=[1], σ=0, etas=[sample_eta1, sample_eta2], u0=u0_, tspan=(-0.1, ind.t[end] + 10))
        push!(list_predicted, predicted)

        # Plot predicted pk
        plot!(plt, saveat, predicted, alpha=0.2, color="#BBBBBB", label="");
    end
    # Plot observed values
    scatter!(plt, ind.t, ind.y, color="red", label="Observed values")

    return list_predicted, saveat, ps, plt
end

function run_chain_fixed_etas(pkmodel::Function, ind::BasicIndividual, I::AbstractMatrix, priors::Dict, etas::AbstractVector, args...; algo=NUTS(0.65), iters::Int=2000, chains::Int=3, sigma=5, kwargs...)
    @model function model_u0_fixed_etas(pkmodel, ind, I, priors, etas, args...; kwargs...)
        u01 ~ priors["u01_prior"]
        u02 ~ priors["u02_prior"]
    
        u0_ = [u01, u02]
    
        predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=etas, u0=u0_, tspan=(-0.1, ind.t[end] + 10), kwargs...)
    
        ind.y ~ MultivariateNormal(vec(predicted), sigma)
    
        return nothing
    end

    # Build model
    model = model_u0_fixed_etas(pkmodel, ind, I, priors, etas);

    # Sample from model
    chain = sample(model, algo, MCMCSerial(), iters, chains; progress=true);
    return chain
end
