using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc.jl"));

pk_model_selection = "mceneny"

if pk_model_selection == "bjorkman"
    include(srcdir("bjorkman.jl"));

    df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);

    pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

    sigma = 5

    sigma_type = "additive";
else
    include(srcdir("mceneny.jl"));

    df = CSV.read(datadir("exp_pro", "variable_times", "mceneny_population_1h.csv"), DataFrame);
    df.ffm = df.weight*(1-0.3);

    pkmodel(args...; kwargs...) = predict_pk_mceneny(args...; kwargs...);

    sigma = 0.17

    sigma_type = "proportional";
end

df_ = df[df.id .== 6, :];  #19, 5, 1, #11, 12, 5, 15, 21, 26


between_dose = 1; #Time between dose for measurments used for MCMC
df_ = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df_);

metadata = eval(Meta.parse(df_[1,:metadata]))

ind, I = individual_from_df(df_);

# Define priors
u0_prior = Truncated(Exponential(10), 0, 60);
#u0_prior = Truncated(Normal(20,10), 0, 60);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plt_u01, plt_u02 = plot_priors_u0(priors);
plt_etas = plot_priors_etas(priors);
plot(plt_u01, plt_u02, plt_etas, layout=(3,1), size = (800, 600))


# Run MCMC
mcmcmodel = model_u0_etas(pkmodel, ind, I, priors; sigma=sigma, sigma_type=sigma_type);
chain_u0_etas = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);
plt = plot(chain_u0_etas)
#save_plots && savefig(plt, plotsdir("chain_multi.png"))
#savefig(plt, plotsdir("chain_multi.png"))

# Sample from chain and recreate curves
#list_predicted, times, ps, plt = sample_posterior(chain, ind, I; n=100, saveat=0.1);
#display(plt)

# Get parameters and modes
#pars = chain.name_map.parameters;

# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;
mode_u01 = mode(round.(chain_u0_etas[:u01].data./round_u0s).*round_u0s);
mode_u02 = mode(round.(chain_u0_etas[:u02].data./round_u0s).*round_u0s);
mode_eta1 = mode(round.(chain_u0_etas[Symbol("etas[1]")].data./round_etas).*round_etas);
mode_eta2 = mode(round.(chain_u0_etas[Symbol("etas[2]")].data./round_etas).*round_etas);

df[1,:metadata]

println("u0s: $([mode_u01, mode_u02])")
println("etas: $([mode_eta1, mode_eta2])")