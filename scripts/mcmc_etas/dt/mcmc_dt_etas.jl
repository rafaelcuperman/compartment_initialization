using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc.jl"));

save_plots = false

pk_model_selection = "bjorkman"

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

df_ = df[df.id .== 1, :];  #19, 5, 1, #11, 12, 5, 15, 21, 26


between_dose = 1; #Time between dose for measurments used for MCMC
df_ = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df_);

metadata = eval(Meta.parse(df_[1,:metadata]))

ind, I = individual_from_df(df_);

# Define priors
#dose_prior = Truncated(MixtureModel(map(u -> Normal(u, 20), I[2]-250:250:I[2]+250)), 0, 3000);
#dose_prior = Truncated(Normal(I[2], 500), 0, 4000);
#time_prior = Truncated(MixtureModel(map(u -> Normal(u, 3), 24:24:72)), 0, 96);
#time_prior = Truncated(Normal(48, 40), 0, 80);
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
save_plots && savefig(plt, plotsdir("priors_informative.png"))

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
mcmcmodel = model_dt_etas(pkmodel, ind, I, priors; sigma=sigma, sigma_type=sigma_type);
#chain_dt = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);
chain_dt = sample(mcmcmodel, MH(), MCMCThreads(), 50000, 3; progress=true);
plt = plot(chain_dt[1:end])
save_plots && savefig(plt, plotsdir("chain_informative.png"))

# Sample from chain and recreate curves
list_predicted, times, ps, plt_restarted, plt_centered = sample_posterior_dt_eta(chain_dt, ind, I; n=100, saveat=0.1, plot_scatter=false);
display(plt_restarted)
display(plt_centered)
save_plots && savefig(plt_centered, plotsdir("prediction_informative.png"))
save_plots && savefig(plt_restarted, plotsdir("prediction_restarted_informative.png"))

df[1,:metadata]