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
u0_prior = Truncated(Exponential(10), 0, 60);
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

# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;
mode_u01 = mode(round.(chain_u0_etas[:u01].data./round_u0s).*round_u0s);
mode_u02 = mode(round.(chain_u0_etas[:u02].data./round_u0s).*round_u0s);
mode_eta1 = mode(round.(chain_u0_etas[Symbol("etas[1]")].data./round_etas).*round_etas);
mode_eta2 = mode(round.(chain_u0_etas[Symbol("etas[2]")].data./round_etas).*round_etas);
pred_etas = [mode_eta1, mode_eta2];

# Get real values of etas and u0s
real_etas = round.(metadata["etas"]./round_etas).*round_etas;
real_u0s = round.(metadata["u0s"]./round_u0s).*round_u0s;
real_dose = metadata["dose"];
real_time = metadata["time"];

println("Real u0s: $(real_u0s). Pred u0s: $([mode_u01, mode_u02])")
println("Real etas: $(real_etas). Pred etas: $(pred_etas)")


########## Predict dose and time based on predicted etas ###########

dose_prior = DiscreteUniform(-1,1)*250 + I[2];
time_prior = Categorical(ones(3)/3).*24;
priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    );

# Plot priors
plt_dose = plot_priors_dose(priors);
plt_time = plot_priors_time(priors);
#plot(plt_dose, plt_time, layout=(2,1), size = (800, 600))

mcmcmodel = model_dt(pkmodel, ind, I, priors, pred_etas; sigma=sigma, sigma_type=sigma_type);
chain_dt = sample(mcmcmodel, MH(), MCMCThreads(), 20000, 3; progress=true);
plt = plot(chain_dt)
save_plots && savefig(plt, plotsdir("chain_dt.png"))

round_time = 24;
round_dose = 250;
pred_time = mode(round.(chain_dt[:t].data./round_time).*round_time)
pred_time = mode(chain_dt[:t])
pred_dose = mode(round.(chain_dt[:D].data./round_dose).*round_dose);

println("Real time: $(real_time). Pred time: $(pred_time)")
println("Real dose: $(real_dose). Pred dose: $(pred_dose)")



########## Run forward u0###########

u01_forward, u02_forward = forward_u0(pkmodel, chain_dt, ind, I, pred_etas; n=1000);

mode_u01_forward = mode(round.(u01_forward./round_u0s).*round_u0s);
mode_u02_forward = mode(round.(u02_forward./round_u0s).*round_u0s);
println("Real u0s: $(real_u0s). Pred u0s: $([mode_u01_forward, mode_u02_forward])")

plt_u01 = density(u01_forward, title="u0[1]", label="Second step", color="blue");
density!(plt_u01, vcat(chain_u0_etas[:u01].data...), label="First step", color="red");
vline!(plt_u01,[real_u0s[1]], label="Real u0[1]", color="black", linestyle=:dash);

plt_u02 = density(u02_forward, title="u0[2]", label="Second step", color="blue");
density!(plt_u02, vcat(chain_u0_etas[:u02].data...), label="First step", color="red");
vline!(plt_u02,[real_u0s[2]], label="Real u0[2]", color="black", linestyle=:dash);

plt = plot(plt_u01, plt_u02, layout=(2,1));
display(plt)
save_plots && savefig(plt, plotsdir("posterior_u0_forward.png"))
