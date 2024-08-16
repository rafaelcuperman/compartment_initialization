using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using GLM

include(srcdir("bjorkman.jl"));
include(srcdir("aux_plots.jl"));

# Boolean to control if plots are saved
save_plots = false;

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = filter(row -> row.id <= 30, df); # First 30 patients

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

mes = []
maes = []
preds = []
observeds = []
ts = []

real_u0s = [];
pred_u0s = [];

for (ix, i) in enumerate(unique(df.id))
    #if ix == 3
    #    break
    #end    

    println("$ix/$(length(unique(df.id)))")

    # Filter patient if
    df_ = filter(row -> row.id == i, df)
    ind, I = individual_from_df(df_);

    # Get real values of u0s
    metadata = eval(Meta.parse(df_[1,:metadata]));
    push!(real_u0s, metadata["u0s"]);

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
save_plots && savefig(plt, plotsdir("naive-goodness-of-fit.png"))

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
save_plots && CSV.write(plotsdir("naive-errors.csv"), df_results);


# Calculate u0s and etas MAE (MAPE is not calculated because there are values=0)
error_u0s = (hcat(real_u0s...) .- zeros(2));

plt = boxplot(error_u0s', labels="", xticks=(1:2, ["u01","u02"]), ylabel="Error (UI/dL)", fillcolor=:lightgray, markercolor=:lightgray)
save_plots && savefig(plt, plotsdir("u0s_errors_$(between_dose)h.png"))

plt = boxplot(abs.(error_u0s)', labels="", xticks=(1:2, ["u01","u02"]), ylabel="Abs 
Error (UI/dL)", fillcolor=:lightgray, markercolor=:lightgray)
save_plots && savefig(plt, plotsdir("u0s_abserrors_$(between_dose)h.png"))

combined_errors = vcat(mean(error_u0s', dims=1), std(error_u0s', dims=1))
combined_errors_abs = vcat(mean(abs.(error_u0s'), dims=1), std(abs.(error_u0s'), dims=1))
combined_errors = vcat(combined_errors, combined_errors_abs)
combined_errors = DataFrame(combined_errors, ["u01", "u02"]);
combined_errors.metric = ["mean error", "std error", "mean abserror", "std abserror"];
save_plots && CSV.write(plotsdir("params_errors_$(between_dose)h.csv"), combined_errors);