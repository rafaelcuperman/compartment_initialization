using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc.jl"));

save_plots = true

pk_model_selection = "mceneny"

if pk_model_selection == "bjorkman"
    include(srcdir("bjorkman.jl"));

    df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);

    pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

    sigma_additive = 5
    sigma_proportional = 0
    sigma = sigma_additive

else
    include(srcdir("mceneny.jl"));

    df = CSV.read(datadir("exp_pro", "variable_times", "mceneny_population_1h.csv"), DataFrame);
    df.ffm = df.weight*(1-0.3);

    pkmodel(args...; kwargs...) = predict_pk_mceneny(args...; kwargs...);

    sigma_additive = 0
    sigma_proportional = 0.17
    sigma = sigma_proportional

end

# Run MCMC for each patient
between_dose = 1; #Time between dose for measurments used for MCMC

# Rounding parameters
round_u0s = 1;
round_etas = 0.1;
round_time = 24;
round_dose = 250;

real_etas = [];
real_u0s = [];
pred_etas = [];
pred_u0s_step1 = [];
pred_u0s_step2 = [];
pred_times = [];
pred_doses = [];

for (ix, i) in enumerate(unique(df.id))
    #if ix == 3
    #    break
    #end    

    println("$ix/$(length(unique(df.id)))")

    # Filter patient i
    df_ = filter(row -> row.id == i, df)
    ind, I = individual_from_df(df_);

    # Get real values of etas and u0s
    metadata = eval(Meta.parse(df_[1,:metadata]));
    push!(real_etas, round.(metadata["etas"]./round_etas).*round_etas);
    push!(real_u0s, round.(metadata["u0s"]./round_u0s).*round_u0s);

    # Filter observations that will be used for MCMC. The rest are used only for evaluation
    df_use = filter(row -> (row.mdv == 1) .| (row.time ∈ 1:between_dose:ind.t[end]), df_);
    ind_use, I_use = individual_from_df(df_use);

    ######### First step: MCMC for etas and u0s #########
    # Define priors
    u0_prior = Truncated(Exponential(10), 0, 60);
    etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
    priors = Dict(
        "u01_prior" => u0_prior,
        "u02_prior" => u0_prior,
        "etas_prior" => etas_prior
        );

    # Run MCMC
    mcmcmodel = model_u0_etas(pkmodel, ind_use, I_use, priors; sigma_additive=sigma_additive, sigma_proportional=sigma_proportional);
    chain_u0_etas = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);

    # Rounding parameters for u0s and etas
    mode_u01 = mode(round.(chain_u0_etas[:u01].data./round_u0s).*round_u0s);
    mode_u02 = mode(round.(chain_u0_etas[:u02].data./round_u0s).*round_u0s);
    mode_eta1 = mode(round.(chain_u0_etas[Symbol("etas[1]")].data./round_etas).*round_etas);
    mode_eta2 = mode(round.(chain_u0_etas[Symbol("etas[2]")].data./round_etas).*round_etas);
    mode_etas = [mode_eta1, mode_eta2]

    #map_chain_u0_etas = maximum_a_posteriori(mcmcmodel).values;
    #mode_u01 = map_chain_u0_etas[:u01]
    #mode_u02 = map_chain_u0_etas[:u02]
    #mode_eta1 = map_chain_u0_etas[Symbol("etas[1]")]
    #mode_eta2 = map_chain_u0_etas[Symbol("etas[2]")]

    # Get predicted values of etas and u0
    push!(pred_etas, mode_etas);
    push!(pred_u0s_step1, [mode_u01, mode_u02]);

    ######### Second step: MCMC dose and time based on predicted etas #########
    # Define priors
    dose_prior = DiscreteUniform(-1,1)*250 + I[2];
    time_prior = Categorical(ones(3)/3).*24;
    priors = Dict(
        "dose_prior" => dose_prior,
        "time_prior" => time_prior,
        );

    # Run MCMC
    mcmcmodel = model_dt(pkmodel, ind_use, I_use, priors, mode_etas; sigma_additive=sigma_additive, sigma_proportional=sigma_proportional);
    chain_dt = sample(mcmcmodel, MH(), MCMCThreads(), 10000, 3; progress=true);

    # Rounding parameters for u0s and etas
    pred_time = mode(round.(chain_dt[:t].data./round_time).*round_time)
    pred_dose = mode(round.(chain_dt[:D].data./round_dose).*round_dose);

    # Get predicted values of dose and time
    push!(pred_times, pred_time);
    push!(pred_doses, pred_dose);

    ######## Run forward with sampled dose and time and fixed etas ########

    u01_forward, u02_forward = forward_u0(pkmodel, chain_dt, ind_use, I_use, mode_etas; n=1000);
    mode_u01_forward = mode(round.(u01_forward./round_u0s).*round_u0s);
    mode_u02_forward = mode(round.(u02_forward./round_u0s).*round_u0s);

    # Get predicted values of  u0
    push!(pred_u0s_step2, [mode_u01_forward, mode_u02_forward]);
end

# Calculate u0s MAE (MAPE is not calculated because there are values=0)
error_u0s = (hcat(real_u0s...) - hcat(pred_u0s_step2...));

plt = boxplot(error_u0s', labels="", xticks=(1:2, ["u01","u02"]), ylabel="Error (UI/dL)", fillcolor=:lightgray, markercolor=:lightgray)
save_plots && savefig(plt, plotsdir("u0s_errors_$(between_dose)h.png"))

plt = boxplot(abs.(error_u0s)', labels="", xticks=(1:2, ["u01","u02"]), ylabel="Abs 
Error (UI/dL)", fillcolor=:lightgray, markercolor=:lightgray)
save_plots && savefig(plt, plotsdir("u0s_abserrors_$(between_dose)h.png"))

combined_errors = hcat(mean(vcat(error_u0s), dims=2), std(vcat(error_u0s), dims=2));
combined_errors_abs = hcat(mean(abs.(vcat(error_u0s)), dims=2), std(abs.(vcat(error_u0s)), dims=2));
combined_errors = hcat(combined_errors, combined_errors_abs);
combined_errors = DataFrame(combined_errors', ["u01", "u02"]);
combined_errors.metric = ["mean error", "std error", "mean abserror", "std abserror"];
println(combined_errors)
save_plots && CSV.write(plotsdir("params_errors_$(between_dose)h.csv"), combined_errors);