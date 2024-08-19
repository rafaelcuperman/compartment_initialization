using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_dt_etas.jl"));
include(srcdir("bjorkman.jl"));

# Boolean to control if plots are saved
save_plots = false;

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = df[df.id .== 1, :];

ind, I = individual_from_df(df);

# Define priors
#dose_prior = Truncated(MixtureModel(map(u -> Normal(u, 10), 1000:250:3000)), 1000, 3000);
dose_prior = Truncated(Normal(I[2], 50), I[2]-250, I[2]+250);
time_prior = Truncated(MixtureModel(map(u -> Normal(u,1), 24:24:72)), 0, 96);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plt = plot_priors_dt(priors);
display(plt)
save_plots && savefig(plt, plotsdir("priors_informative.png"))

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
chain = run_chain(pkmodel, ind, I, priors; algo=NUTS(0.65), iters=2000, chains=3, sigma=5)
plt = plot(chain)
save_plots && savefig(plt, plotsdir("chain_informative.png"))

# Sample from chain and recreate curves
list_predicted, times, ps, plt_restarted, plt_centered = sample_posterior(chain, ind, I; n=100, saveat=0.1);
display(plt_restarted)
display(plt_centered)
save_plots && savefig(plt_centered, plotsdir("prediction_informative.png"))
save_plots && savefig(plt_restarted, plotsdir("prediction_restarted_informative.png"))

df[1,:metadata]