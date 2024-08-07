using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using GLM
using Plots
using Printf

function goodness_of_fit(preds, observed)
    # Plot points and x=y line
    max_val=ceil(max(maximum(observed), maximum(preds)));
    plt = scatter(vcat(preds), vcat(observed), markersize=2, color="black", xlabel="Predicted", ylabel="Observed", label="")
    plot!(plt, [0, max_val], [0, max_val], color="red", linestyle=:dash, label="Identity line")

    # Fit linear regression to points and calculate r2
    lr = GLM.lm(GLM.@formula(y ~ X), DataFrame(X=vcat(preds), y= vcat(observed)));
    rsquared = GLM.r2(lr);

    # Plot regression line
    regression = GLM.predict(lr, DataFrame(X=[0, max_val]))
    Plots.plot!(plt, [0, max_val], regression, color = "blue", title = "R2 = $(@sprintf("%.2f", rsquared))", label="Regression line")
    return plt, rsquared
end