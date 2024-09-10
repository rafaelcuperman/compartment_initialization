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

# Run MCMC for each patient
between_dose = 1; #Time between dose for measurments used for MCMC

# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;

real_etas = [];
real_u0s = [];
pred_etas = [];
pred_u0s_step1 = [];
pred_u0s_step2 = [];

for (ix, i) in enumerate(unique(df.id))
    if ix == 3
        break
    end    

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
    mcmcmodel = model_u01_etas(pkmodel, ind_use, I_use, priors; sigma=sigma, sigma_type=sigma_type);
    chain_u0_etas = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);

    # Get predicted modes. The values are rounded to the nearest round_u0s and round_etas to get the modes
    mode_u01 = mode(round.(chain_u0_etas[:u01].data./round_u0s).*round_u0s);
    mode_u02 = 0;
    mode_eta1 = mode(round.(chain_u0_etas[Symbol("etas[1]")].data./round_etas).*round_etas);
    mode_eta2 = mode(round.(chain_u0_etas[Symbol("etas[2]")].data./round_etas).*round_etas);

    # Get predicted values of etas
    push!(pred_etas, [mode_eta1, mode_eta2]);
    push!(pred_u0s_step1, [mode_u01, mode_u02]);

    ######### Second step: fix etas and do MCMC only for u0s #########
    # Define priors
    u01_prior = Truncated(Normal(mode_u01, 10), 0, 60);
    u02_prior = Truncated(Normal(mode_u02, 10), 0, 1);
    priors = Dict(
        "u01_prior" => u01_prior,
        "u02_prior" => u02_prior,
        );

    # Run MCMC
    etas = [mode_eta1, mode_eta2]
    mcmcmodel = model_u0(pkmodel, ind_use, I_use, priors, etas; sigma=sigma, sigma_type=sigma_type);
    chain_u0 = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);

    mode2_u01 = mode(round.(chain_u0[:u01].data./round_u0s).*round_u0s);
    mode2_u02 = mode(round.(chain_u0[:u02].data./round_u0s).*round_u0s);

    # Get predicted values of u0s
    push!(pred_u0s_step2, [mode2_u01, mode2_u02]);
end


# Calculate u0s and etas MAE (MAPE is not calculated because there are values=0)
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