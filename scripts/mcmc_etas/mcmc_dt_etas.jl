using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_dt_etas.jl"));
include(srcdir("bjorkman.jl"));

# Read data
sigma = 5;
boolean_etas = "y";
df = CSV.read(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), DataFrame);

ind, I = individual_from_df(df);

# Define priors
#dose_prior = Truncated(Normal(1750, 1000), 1000, 3000);
#time_prior = Truncated(Normal(12, 10), 6, 36);

dose_prior = Truncated(MixtureModel(map(u -> Normal(u, 10), 1000:250:3000)), 1000, 3000);
time_prior = Truncated(MixtureModel(map(u -> Normal(u, 0.5), 0:6:36)), 0, 36);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plot_priors_dt(priors);

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
chain = run_chain(pkmodel, ind, I, priors; algo=NUTS(0.65), iters=3000, chains=3, sigma=5)
plot(chain)

#savefig(plt, plotsdir("chain_informative.png"))

# Sample from chain and recreate curves
list_predicted, times, ps, plt_restarted, plt_centered = sample_posterior(chain, ind, I; n=100, saveat=0.1);
display(plt_restarted)
display(plt_centered)

#savefig(plt_centered, plotsdir("prediction_informative.png"))