using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_u0_etas.jl"));
include(srcdir("bjorkman.jl"));

# Read data
df = CSV.read(datadir("exp_pro", "bjorkman_population_1h.csv"), DataFrame);

# Define priors
u0_prior = Truncated(Normal(20, 20), 0, 60);
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

# Run MCMC for each patient
mapes = []
maes = []
average_preds = []
observeds = []
for (ix, i) in enumerate(unique(df.id))
    if ix == 10
        break
    end    

    println("$i/$(length(unique(df.id)))")

    df_ = filter(row -> row.id == i, df)
    ind, I = individual_from_df(df_);

    # Run MCMC
    chain = run_chain(pkmodel, ind, I, priors; algo=NUTS(0.65), iters=1000, chains=1, sigma=5)

    # Sample from chain and recreate curves
    list_predicted, times, _ = sample_posterior(chain, ind, I; n=100);

    # Take average prediction for all times
    average_vector = mean.(reduce((x, y) -> x .+ y, list_predicted) ./ length(list_predicted));

    # Get the indices of the times where there are observations
    indices = vcat([findall(y -> y == element, times) for element in ind.t]...);

    # Get the average prediction where there are observations
    average_vector = average_vector[indices];
    push!(average_preds, average_vector)

    # Save observed values
    push!(observeds, ind.y)

    # Calculate MAPE
    mape = mean(abs.((ind.y .- average_vector) ./ ind.y))
    push!(mapes, mape)

    # Calculate MAE
    mae = mean(abs.(ind.y .- average_vector))
    push!(maes, mae)
end

plt = scatter(vcat(average_preds...), vcat(observeds...), markersize=2, color="black",  label="", xlabel="Predicted", ylabel="Observed")

max_r2=ceil(max(maximum(maximum(observeds)), maximum(maximum(average_preds))));
plot!(plt, [0, max_r2], [0, max_r2], label="", color="red", )


"""
i = 2
# Run MCMC
chain = run_chain(pkmodel, inds[i], Is[i], priors; algo=NUTS(0.65), iters=2000, chains=3, sigma=5)
plot(chain)

# Sample from chain and recreate curves
list_predicted, times, plt = sample_posterior(chain, inds[i], Is[i]; n=100);
display(plt)

# Take average prediction for all times
average_vector = mean.(reduce((x, y) -> x .+ y, list_predicted) ./ length(list_predicted));

# Get the indices of the times where there are observations
indices = vcat([findall(y -> y == element, times) for element in inds[i].t]...);

# Get the average prediction where there are observations
average_vector = average_vector[indices];

mean(abs.((inds[i].y .- average_vector) ./ inds[i].y))

plt = scatter(inds[i].t,  inds[i].y)

plot!(plt, times, average_vector)
display(plt)

scatter(inds[i].t, inds[i].y)
"""