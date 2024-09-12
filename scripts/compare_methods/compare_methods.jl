using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV

save_plots = true

pk_model_selection = "bjorkman_5" #"bjorkman_5"

# Naive
df_naive = CSV.read(plotsdir(pk_model_selection, "naive", "params_errors.csv"), DataFrame);

# U0 directly
df_u0 = CSV.read(plotsdir(pk_model_selection, "mcmc_u0", "1h", "params_errors_1h.csv"), DataFrame);
#df_u0 = CSV.read(plotsdir(pk_model_selection, "mcmc_u0", "1h", "etas_errors_1h_0u0s.csv"), DataFrame);

# U0 and etas, then fix etas and predidct u0s
df_2step = CSV.read(plotsdir(pk_model_selection, "mcmc_u0", "1h", "two_step", "params_errors_1h.csv"), DataFrame);

# U0 and etas, then fix etas and predict Dt, then forward pass to predict u0s
df_u0_dt_mean = CSV.read(plotsdir(pk_model_selection, "mcmc_u0_dt", "1h", "params_errors_1h_mean.csv"), DataFrame);

# U0 and etas, then fix etas and predict Dt, then forward pass to predict u0s
df_u0_dt_median = CSV.read(plotsdir(pk_model_selection, "mcmc_u0_dt", "1h", "params_errors_1h_median.csv"), DataFrame);

# U0 and etas, then fix etas and predict Dt, then forward pass to predict u0s
df_u0_dt_mode = CSV.read(plotsdir(pk_model_selection, "mcmc_u0_dt", "1h", "params_errors_1h_mode.csv"), DataFrame);

# Perfect with mean u0
#df_perfect_mean = CSV.read(plotsdir(pk_model_selection, "perfect", "params_errors_mean.csv"), DataFrame);

# Perfect with median u0
#df_perfect_median = CSV.read(plotsdir(pk_model_selection, "perfect", "params_errors_median.csv"), DataFrame);

# Perfect with mode u0
#df_perfect_mode = CSV.read(plotsdir(pk_model_selection, "perfect", "params_errors_mode.csv"), DataFrame);


# Functions to prepare dat and plot
function join_metrics(parameter)
    metrics_df = copy(df_naive[:,[parameter, "metric"]]);

    metrics_df = innerjoin(metrics_df, df_u0[:,[parameter, "metric"]], on="metric", makeunique=true);
    rename!(metrics_df, parameter => "naive", string(parameter, "_1") => "u0s");

    metrics_df = innerjoin(metrics_df, df_2step[:,[parameter, "metric"]], on="metric", makeunique=true);
    rename!(metrics_df, parameter => "etas_u0s");

    metrics_df = innerjoin(metrics_df, df_u0_dt_mean[:,[parameter, "metric"]], on="metric", makeunique=true);
    rename!(metrics_df, parameter => "etas_dt_mean");

    metrics_df = innerjoin(metrics_df, df_u0_dt_median[:,[parameter, "metric"]], on="metric", makeunique=true);
    rename!(metrics_df, parameter => "etas_dt_median");

    metrics_df = innerjoin(metrics_df, df_u0_dt_mode[:,[parameter, "metric"]], on="metric", makeunique=true);
    rename!(metrics_df, parameter => "etas_dt_mode");

    #metrics_df = innerjoin(metrics_df, df_perfect_mean[:,[parameter, "metric"]], on="metric", makeunique=true);
    #rename!(metrics_df, parameter => "perfect_mean");

    #metrics_df = innerjoin(metrics_df, df_perfect_median[:,[parameter, "metric"]], on="metric", makeunique=true);
    #rename!(metrics_df, parameter => "perfect_median");

    #metrics_df = innerjoin(metrics_df, df_perfect_mode[:,[parameter, "metric"]], on="metric", makeunique=true);
    #rename!(metrics_df, parameter => "perfect_mode");

    return metrics_df
end

function plot_metric(df, metric_name; plottype="values", ylim=nothing)
    values = df[df[:, "metric"] .== metric_name, Not("metric")]
    
    col_names = filter!(x -> x != "metric", names(df))

    if plottype == "errors"
        error_name = string("std ", split(metric_name)[end])
        errors = df[df[:, "metric"] .== error_name, Not("metric")]

        ylim = isnothing(ylim) ? (minimum(values[1,:])-minimum(errors[1,:])-2, maximum(values[1,:]) + maximum(errors[1,:])+2) : ylim
        plt = scatter(collect(values[1,:]), 
                yerr=collect(errors[1,:]), 
                label=nothing, 
                xlim=(0, length(col_names)+1), 
                xticks=(1:length(col_names), col_names), 
                xrotation=45,
                ylim=ylim, 
                ylabel=metric_name
                )
        hline!(plt, [0], label=nothing, color="black")
    else
        ylim = isnothing(ylim) ? (minimum(values[1,:])-2, maximum(values[1,:])+2) : ylim
        plt = plot(collect(values[1,:]),
            label=nothing, 
            xlim=(0, length(col_names)+1),
            xticks=(1:length(col_names), col_names),
            xrotation = 45,
            ylim = ylim,
            ylabel=metric_name,
            markersize=5,
            marker=:circle)
        hline!(plt, [0], label=nothing, color="black")
    end
    return plt
end

# Plot errors for etas
metrics_etas = df_u0[:, ["eta1", "eta2", "metric"]];

plt_error_eta = plot_metric(metrics_etas, "mean error"; plottype="values", ylim=(-0.3,0.3))

save_plots && savefig(plt_error_eta, plotsdir("error_eta_.png"))

plt_abserror_eta = plot_metric(metrics_etas, "mean abserror"; plottype="values", ylim=(0,0.2))

save_plots && savefig(plt_abserror_eta, plotsdir("abserror_eta.png"))

#plot(plt_error_eta, plt_abserror_eta, layout=(2,1))

# Plot errors for u0s
metrics_u01 = join_metrics("u01");
metrics_u02 = join_metrics("u02");

plt = plot_metric(metrics_u01, "mean error", plottype="values")
save_plots && savefig(plt, plotsdir("error_u01.png"))

plt = plot_metric(metrics_u01, "mean abserror", plottype="values")
save_plots && savefig(plt, plotsdir("abserror_u01.png"))


plt = plot_metric(metrics_u02, "mean error", plottype="values")
save_plots && savefig(plt, plotsdir("error_u02.png"))

plt = plot_metric(metrics_u02, "mean abserror", plottype="values")
save_plots && savefig(plt, plotsdir("abserror_u02.png"))
