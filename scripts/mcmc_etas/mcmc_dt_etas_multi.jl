using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using GLM

include(srcdir("mcmc_dt_etas.jl"));
include(srcdir("bjorkman.jl"));
include(srcdir("aux_plots.jl"));

# Read data
df = CSV.read(datadir("exp_pro", "bjorkman_population_1h.csv"), DataFrame);

# Define priors
dose_prior = Truncated(MixtureModel(map(u -> Normal(u, 10), 1000:250:3000)), 1000, 3000);
time_prior = Truncated(MixtureModel(map(u -> Normal(u, 0.5), 0:6:36)), 0, 36);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plot_priors_dt(priors);

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC for each patient
between_dose=1; #Time between dose for measurments used for MCMC

mes = []
maes = []
average_preds = []
observeds = []
ts = []
for (ix, i) in enumerate(unique(df.id))
    #if ix == 3
    #    break
    #end    

    println("$ix/$(length(unique(df.id)))")

    # Filter patient if
    df_ = filter(row -> row.id == i, df)
    ind, I = individual_from_df(df_);

    # Filter observations that will be used for MCMC. The rest are used only for evaluation
    df_use = filter(row -> (row.mdv == 1) .| (row.time ∈ 1:between_dose:ind.t[end]), df_);
    ind_use, I_use = individual_from_df(df_use);

    # Run MCMC
    chain = run_chain(pkmodel, ind_use, I_use, priors; algo=NUTS(0.65), iters=2000, chains=1, sigma=5)

    # Sample from chain and recreate n curves
    list_predicted, times, ps, _, _ = sample_posterior(chain, ind, I; n=100, saveat=0.1);

    # Center each prediction to t0 and align with measurements
    list_predicted_centered = []
    for jx in eachindex(list_predicted)
        t = times .- round(ps[jx][2]) # Subtract predicted time of first dose
        indices = vcat([findall(y -> y == element, t) for element in ind.t]...); # Get indices where observations are
        push!(list_predicted_centered, list_predicted[jx][indices]) # Get predictions where observations are
    end

    # Take average prediction of all times across the n curves
    average_vector = vec(mean.(reduce((x, y) -> x .+ y, list_predicted_centered) ./ length(list_predicted_centered)));

    # Save time
    ts = ind.t

    # Save average predictions
    push!(average_preds, average_vector)

    # Save observed values
    push!(observeds, ind.y)

    # Calculate ME
    me = mean(ind.y .- average_vector)
    push!(mes, me)

    # Calculate MAE
    mae = mean(abs.(ind.y .- average_vector))
    push!(maes, mae)
end

# Plot goodness of fit
plt, rsquared = goodness_of_fit(vcat(average_preds...), vcat(observeds...));
display(plt)
savefig(plt, plotsdir("goodness-of-fit_$(between_dose)h.png"))

# Display average MAE and ME for all patients
mean_mae = mean(maes);
std_mae = std(maes);

mean_me = mean(mes);
std_me = std(mes);

df_results = DataFrame(mean_mae=mean_mae,
                        std_mae=std_mae,
                        mean_me=mean_me,
                        std_me=std_me
                        );
println(df_results)
CSV.write(plotsdir("errors_$(between_dose)h.csv"), df_results);


# Add the lists element-wise using list comprehensions
errors = [observeds[i] .- average_preds[i] for i in eachindex(observeds)];
mean_errors = mean(errors);
std_errors = std(errors);
plt = plot(ts, mean_errors, ribbon=(std_errors, std_errors), xlabel="Time", ylabel="Error", label="")
savefig(plt, plotsdir("errors_$(between_dose)h.png"))
