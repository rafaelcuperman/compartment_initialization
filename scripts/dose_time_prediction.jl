using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using ArviZ
using Random

include(srcdir("mcmc.jl"));
include(srcdir("metrics.jl"));
include(srcdir("utils.jl"));

include(srcdir("bjorkman.jl")); # Model that will be used to model the data and make predictions
type_prior = "continuous";
times = [24, 48, 72];
metric = "joint"; # joint, ml_post, ml_prior, ml_is, loo, waic

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
#df.ffm = df.weight*(1-0.3);

data = df[df.id .== 5, :]; #5, 19

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

    return norm_metric, plt_metric
end;

norm_metric, plt_metric = dose_time_prediction(type_prior, times, metric, data)