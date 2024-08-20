using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_u0_etas.jl"));
include(srcdir("bjorkman.jl"));

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = df[df.id .== 29, :];

between_dose = 1; #Time between dose for measurments used for MCMC
df = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df);

ind, I = individual_from_df(df);

# Define priors
#u0_prior = Truncated(Normal(10, 20), 0, 60);
u0_prior = Truncated(Exponential(10), 0, 60);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plt = plot_priors_u0(priors);
display(plt)

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
chain = run_chain(pkmodel, ind, I, priors; algo=NUTS(0.65), iters=2000, chains=3, sigma=5)
plt = plot(chain)
display(plt)
#savefig(plt, plotsdir("chain_multi.png"))

# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;

# Get parameters and modes
mode_u01 = mode(round.(chain[:u01].data./round_u0s).*round_u0s);
mode_u02 = mode(round.(chain[:u02].data./round_u0s).*round_u0s);
mode_eta1 = mode(round.(chain[Symbol("etas[1]")].data./round_etas).*round_etas);
mode_eta2 = mode(round.(chain[Symbol("etas[2]")].data./round_etas).*round_etas);

metadata = eval(Meta.parse(df[1,:metadata]));
println("etas real: $(round.(metadata["etas"]./round_etas).*round_etas)")
println("etas predicted: $([mode_eta1, mode_eta2])")


######### Second step: fix etas and do MCMC only for u0s #########
# Define priors
u01_prior = Truncated(Normal(mode_u01, 10), 0, 60);
u02_prior = Truncated(Normal(mode_u02, 10), 0, 60);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "u01_prior" => u01_prior,
    "u02_prior" => u02_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plt = plot_priors_u0(priors);
display(plt)

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
etas = [mode_eta1, mode_eta2]
chain2 = run_chain_fixed_etas(pkmodel, ind, I, priors, etas; algo=NUTS(0.65), iters=2000, chains=3, sigma=5)
plt2 = plot(chain2)
display(plt2)

mode2_u01 = mode(round.(chain2[:u01].data./round_u0s).*round_u0s);
mode2_u02 = mode(round.(chain2[:u02].data./round_u0s).*round_u0s);

println("u0s real: $(metadata["u0s"])")
println("u0s step 1: $([mode_u01, mode_u02])")
println("u0s step 2: $([mode2_u01, mode2_u02])")