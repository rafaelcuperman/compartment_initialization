using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV

include(srcdir("bjorkman.jl"));

# Read data
sigma = 5;
boolean_etas = "y";
df = CSV.read(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), DataFrame);
ind, I = individual_from_df(df_);

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

saveat = collect(0:0.1:ind.t[end] + 10);
pred = vec(pkmodel(ind, I, saveat; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, ind.t[end] + 10)));

# Plot predicted pk
plt = plot(saveat, pred, color="black", label="");
scatter!(plt, ind.t, ind.y, color="red", label="Observed values")
display(plt)

savefig(plt, plotsdir("naive-predictions.png"))