using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_u0_etas.jl"));
include(srcdir("bjorkman.jl"));

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = df[df.id .== 1, :];

ind, I = individual_from_df(df);

# Define priors
#u0_prior = Truncated(Normal(0, 20), 0, 60);
u0_prior = Truncated(Exponential(10), 0, 60);
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
list_predicted, times, ps, plt = sample_posterior(chain, ind, I; n=100, saveat=0.1);
display(plt)

# Get parameters and modes
pars = chain.name_map.parameters;
modes = Dict(string(param) => mode(chain[param][:]) for param in pars);