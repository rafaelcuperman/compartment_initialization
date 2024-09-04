using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc.jl"));
include(srcdir("bjorkman.jl"));

# Boolean to control if plots are saved
save_plots = false;

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = df[df.id .== 1, :];

metadata = eval(Meta.parse(df_[1,:metadata]));

sigma = 5

ind, I = individual_from_df(df);

# Define priors
#dose_prior = Truncated(MixtureModel(map(u -> Normal(u, 10), 1000:250:3000)), 1000, 3000);
dose_prior = Truncated(Normal(I[2], 50), I[2]-250, I[2]+250);
time_prior = Truncated(MixtureModel(map(u -> Normal(u,1), 24:24:72)), 0, 96);
dose_prior = DiscreteUniform(-1,1)*250 + I[2];
time_prior = Categorical(ones(3)/3).*24;
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plt_etas = plot_priors_etas(priors);
plt_dose = plot_priors_dose(priors);
plt_time = plot_priors_time(priors);
plt = plot(plt_dose, plt_time, plt_etas, layout=(3,1), size = (800, 600))
#display(plt)
#save_plots && savefig(plt, plotsdir("priors_informative.png"))

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
mcmcmodel = model_dt_etas(pkmodel, ind, I, priors; sigma=sigma, sigma_type="additive");
chain_dt = sample(mcmcmodel, MH(), MCMCThreads(), 10000, 3; progress=true);
plt = plot(chain_dt)
save_plots && savefig(plt, plotsdir("chain_informative.png"))

# Sample from chain and recreate curves
list_predicted, times, ps, plt_restarted, plt_centered = sample_posterior_dt_eta(chain_dt, ind, I; n=100, saveat=0.1);
display(plt_restarted)
display(plt_centered)
save_plots && savefig(plt_centered, plotsdir("prediction_informative.png"))
save_plots && savefig(plt_restarted, plotsdir("prediction_restarted_informative.png"))

df[1,:metadata]