using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_u0_etas.jl"));
include(srcdir("bjorkman.jl"));

# Read data
sigma = 5;
boolean_etas = "n";
df = CSV.read(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), DataFrame);

ind, I = individual_from_df(df);

# Define priors
u0_prior = Truncated(Normal(20, 20), 0, 60);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plot_priors_u0(priors);

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
chain = run_chain(pkmodel, ind, I, priors; algo=NUTS(0.65), iters=2000, chains=3, sigma=5)
plot(chain)

# Sample from chain and recreate curves
list_predicted, times, plt = sample_posterior(chain, ind, I; n=100, saveat=0.1);
display(plt)