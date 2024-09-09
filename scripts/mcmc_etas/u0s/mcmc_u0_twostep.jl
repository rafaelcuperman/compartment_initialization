using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

save_plots = false

include(srcdir("mcmc.jl"));

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
u0_prior = Truncated(Exponential(10), 0, 60);
#u0_prior = Truncated(Normal(10,20), 0, 60);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
#plt_u01, plt_u02 = plot_priors_u0(priors);
#plt_etas = plot_priors_etas(priors);
#plot(plt_u01, plt_u02, plt_etas, layout=(3,1), size = (800, 600))

# Run MCMC
mcmcmodel = model_u0_etas(pkmodel, ind, I, priors; sigma=sigma, sigma_type=sigma_type);
chain_u0_etas = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);
plt = plot(chain_u0_etas)
save_plots && savefig(plt, plotsdir("chain_mcmc_u0_$(between_dose)h.png"))


# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;

# Get parameters and modes
mode_u01 = mode(round.(chain_u0_etas[:u01].data./round_u0s).*round_u0s);
mode_u02 = mode(round.(chain_u0_etas[:u02].data./round_u0s).*round_u0s);
mode_eta1 = mode(round.(chain_u0_etas[Symbol("etas[1]")].data./round_etas).*round_etas);
mode_eta2 = mode(round.(chain_u0_etas[Symbol("etas[2]")].data./round_etas).*round_etas);

println("etas real: $(round.(metadata["etas"]./round_etas).*round_etas)")
println("etas predicted: $([mode_eta1, mode_eta2])")


######### Second step: fix etas and do MCMC only for u0s #########
# Define priors
u01_prior = Truncated(Normal(mode_u01, 10), 0, 60);
u02_prior = Truncated(Normal(mode_u02, 10), 0, 60);
priors = Dict(
    "u01_prior" => u01_prior,
    "u02_prior" => u02_prior,
    );

# Run MCMC
etas = [mode_eta1, mode_eta2]
mcmcmodel = model_u0(pkmodel, ind, I, priors, etas; sigma=sigma, sigma_type=sigma_type);
chain_u0 = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);
plt = plot(chain_u0)
save_plots && savefig(plt, plotsdir("chain_mcmc_u0_twostep_$(between_dose)h.png"))

mode2_u01 = mode(round.(chain_u0[:u01].data./round_u0s).*round_u0s);
mode2_u02 = mode(round.(chain_u0[:u02].data./round_u0s).*round_u0s);

real_u0s = metadata["u0s"];

println("u0s real: $(real_u0s)")
println("u0s step 1: $([mode_u01, mode_u02])")
println("u0s step 2: $([mode2_u01, mode2_u02])")

plt_u01 = density(vcat(chain_u0[:u01].data...), title="u0[1]", label="Second step", color="blue");
density!(plt_u01, vcat(chain_u0_etas[:u01].data...), label="First step", color="red");
vline!(plt_u01, [real_u0s[1]], label="Real u0[1]", color="black", linestyle=:dash);

plt_u02 = density(vcat(chain_u0[:u02].data...), title="u0[2]", label="Second step", color="blue");
density!(plt_u02, vcat(chain_u0_etas[:u02].data...), label="First step", color="red");
vline!(plt_u02, [real_u0s[2]], label="Real u0[2]", color="black", linestyle=:dash);

plt = plot(plt_u01, plt_u02, layout=(2,1));
display(plt)
save_plots && savefig(plt, plotsdir("posterior_u0_forward.png"))
