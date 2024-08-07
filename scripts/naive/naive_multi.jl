using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using GLM

include(srcdir("bjorkman.jl"));
include(srcdir("aux_plots.jl"));

# Read data
df = CSV.read(datadir("exp_pro", "bjorkman_population_1h.csv"), DataFrame);

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

mes = []
maes = []
preds = []
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

    # Run pk model
    pred = vec(pkmodel(ind, I, ind.t; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, ind.t[end] + 10)))

    # Save time
    ts = ind.t

    # Save average predictions
    push!(preds, pred)

    # Save observed values
    push!(observeds, ind.y)

    # Calculate ME
    me = mean(ind.y .- pred)
    push!(mes, me)

    # Calculate MAE
    mae = mean(abs.(ind.y .- pred))
    push!(maes, mae)
end


# Plot goodness of fit
plt, rsquared = goodness_of_fit(vcat(average_preds...), vcat(observeds...));
display(plt)
savefig(plt, datadir("sims", "naive-goodness-of-fit.png"))

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
CSV.write(datadir("sims", "naive-errors.csv"), df_results);


# Add the lists element-wise using list comprehensions
errors = [observeds[i] .- preds[i] for i in eachindex(observeds)];
mean_errors = mean(errors);
std_errors = std(errors);
plt = plot(ts, mean_errors, ribbon=(std_errors, std_errors), xlabel="Time", ylabel="Error", label="")
savefig(plt, datadir("sims", "naive-errors.png"))
