using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using ArviZ
using Random
using StatsBase

include(srcdir("mcmc.jl"));
include(srcdir("metrics.jl"));
include(srcdir("utils.jl"));

"""
Run dose amount and time prediction algorithm

# Arguments
- `type_prior`: Type of prior for the dose amount. "continuous" or "discrete". A normal prior is used centered in the weight-based dose
- `times`: Array of feasible times for the initial dose
- `metric`: Metric to be used to compare models: "joint", "ml_post", "ml_prior", "ml_is", "loo", or "waic"
- `data`: Dataframe with data to be used to build the model
"""
function dose_time_prediction(type_prior, times, metric, data)
    pkmodel(args...; kwargs...) = predict_pk(args...; kwargs...);

    # Build intervention matrix
    ind, I = individual_from_df_general(data);
    
    # Define priors
    if type_prior == "discrete"
        # Build discrete normal prior based on categorical distributions
        possible_doses = collect(250:250:4000);
        dist = Normal(I[2], 500);
        probs = [pdf(dist, i) for i in possible_doses];
        probs = probs/sum(probs);
        dose_prior = Categorical(probs).*250;
        burnin = 20000
    elseif type_prior == "continuous"  
        dose_prior = Truncated(Normal(I[2], 500), 0, 4000);
        burnin = 1
    else
        throw(ExceptionError("Unknown type_prior"))
    end;
    etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
    priors = Dict(
        "dose_prior" => dose_prior,
        "etas_prior" => etas_prior
        );
    
    # Run MCMC
    chains = [];
    models = [];
    for (ix, t) in enumerate(times)
        # Run MCMC
        println("Running chain $ix/$(length(times)). Using t=$t")
        mcmcmodel = model_dose_etas(pkmodel, ind, I, priors, t; sigma_additive=sigma_additive, sigma_proportional=sigma_proportional);
        if type_prior == "discrete"
            chain = sample(mcmcmodel, MH(), MCMCThreads(), 100000, 3; progress=true);
        else
            chain = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);
        end
        push!(chains, chain)
        push!(models, mcmcmodel)
    end
    println("Models built")
    
    if metric == "joint"
        norm_metric, plt_metric = joint_posterior_mean(chains, models);
    elseif metric == "ml_post"
        norm_metric, plt_metric = ml_post(chains, models);
    elseif metric == "ml_prior"
        norm_metric, plt_metric = marginal_likelihood_approx(chains, models);
    elseif metric == "ml_is"
        norm_metric, plt_metric = importance_sampling(chains, models);
    elseif metric == "loo"
        norm_metric, _, plt_metric, _ = loo_waic(chains, models);
    elseif metric == "waic"
        _, norm_metric, _, plt_metric = loo_waic(chains, models);
    else
        throw(ExceptionError("Unknown metric. Choose one of joint, ml_post, ml_prior, ml_is, loo, waic"))
    end;
    
    display(plt_metric)

    return norm_metric, plt_metric, chains
end;

"""
Sample dose, time, and etas from resulting model

# Arguments
- `times`: Array of feasible times for the initial dose
- `chains`: Array of chains of posterior distributions obtained by the model
- `prob_models`: Array of probabilities obtained by the algorithm
- `round_dose`: The obtained dose will be rounded to the nearest multiple of this value
- `round_etas`: The obtained etas will be rounded to the nearest multiple of this value
"""
function sample_dose_time(times, chains, prob_models; round_dose=250, round_etas=0.1)
    # Pick a model based on the calculated probabilities of models
    time = sample(times, Weights(prob_models));

    # Retrieve the posterior distributions of dose and etas based on the chosen model
    posterior = Array(chains[findfirst(x -> x == time, times)]);

    # Pick random dose and eta from the posterior distributions
    random_indices = sample(1:size(posterior, 1))
    dose_etas = posterior[random_indices, :]

    # Round dose and etas
    dose = round(dose_etas[1]/round_dose)*round_dose
    etas = round.(dose_etas[2:end]./round_etas).*round_etas

    return time, dose, etas
end
