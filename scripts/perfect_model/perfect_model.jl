### This model uses the real etas, dose, and time and forwards the ODE 100 times to get the predicted u0s. Then, the mean/median/mode of the u0s of the 100 runs is taken as the final prediction

using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc.jl"));

save_plots = true

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

round_u0s = 1;
real_u0s = [];
pred_u0s = [];
for (ix, i) in enumerate(unique(df.id))
    #if ix == 3
    #    break
    #end    

    println("$ix/$(length(unique(df.id)))")

    # Filter patient i
    df_ = filter(row -> row.id == i, df)
    ind, I = individual_from_df(df_);

    # Get real values
    metadata = eval(Meta.parse(df_[1,:metadata]));
    push!(real_u0s, round.(metadata["u0s"]./round_u0s).*round_u0s);

    real_etas = metadata["etas"]
    real_dose = metadata["dose"]
    real_time = metadata["time"]

    # Regenerate initial dose
    I_ = copy(I);
    I_ = vcat([0. 0. 0. 0.], I_);

    # Shift all the dosing times by predicted time except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ real_time;
    I_[1,2] = real_dose;
    I_[1,3] = real_dose*60;
    I_[1,4] = 1/60;

    u01_forward = []
    u02_forward = []
    for i in 1:100
        u0_forward = pkmodel(ind, I_, [real_time]; save_idxs=[1,2], σ=sigma, etas=real_etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ real_time))
        
        push!(u01_forward, u0_forward[1])
        push!(u02_forward, u0_forward[2])
    end   
    mode_u01_forward = median(round.(u01_forward./round_u0s).*round_u0s);
    mode_u02_forward = median(round.(u02_forward./round_u0s).*round_u0s);

    push!(pred_u0s, [mode_u01_forward, mode_u02_forward])

end
# Calculate u0s MAE (MAPE is not calculated because there are values=0)
error_u0s = (hcat(real_u0s...) - hcat(pred_u0s...));

plt = boxplot(error_u0s', labels="", xticks=(1:2, ["u01","u02"]), ylabel="Error (UI/dL)", fillcolor=:lightgray, markercolor=:lightgray)
save_plots && savefig(plt, plotsdir("u0s_errors_median.png"))

plt = boxplot(abs.(error_u0s)', labels="", xticks=(1:2, ["u01","u02"]), ylabel="Abs 
Error (UI/dL)", fillcolor=:lightgray, markercolor=:lightgray)
save_plots && savefig(plt, plotsdir("u0s_abserrors_median.png"))

combined_errors = vcat(mean(error_u0s', dims=1), std(error_u0s', dims=1))
combined_errors_abs = vcat(mean(abs.(error_u0s'), dims=1), std(abs.(error_u0s'), dims=1))
combined_errors = vcat(combined_errors, combined_errors_abs)
combined_errors = DataFrame(combined_errors, ["u01", "u02"]);
combined_errors.metric = ["mean error", "std error", "mean abserror", "std abserror"];
save_plots && CSV.write(plotsdir("params_errors_median.csv"), combined_errors);