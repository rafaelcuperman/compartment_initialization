using DrWatson
@quickactivate "compartment-initialization"

using Turing
using DataFrames
using CSV
using Plots
using StatsPlots

include(srcdir("bjorkman.jl"));

sigma = 5
boolean_etas = "y"

# Read data
df = CSV.read(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), DataFrame);

data = Float64.(df[2:end, :dv]);
times = Float64.(df[2:end, :time]);
last_time = maximum(df[!, :time]);

age = df[1, :age];
weight = df[1, :weight];

# Reconstruct dosing matrix
I = reshape(Float64.(collect(df[1, [:time, :amt, :rate, :duration]])),1,4)

scatter(times, data, color="red", label="Observed values")

# Define and plot priors
u0_prior = Truncated(Normal(20, 20), 0, 60);
x = range(0, stop=80, length=1000);
plt_u0 = plot(x, pdf.(u0_prior, x), title="Sigma prior", label="", yticks=nothing);

etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
x = range(-3, stop=3, length=1000);
y = range(-3, stop=3, length=1000);
X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y];
Z = [pdf(etas_prior, [X[i, j], Y[i, j]]) for i in 1:size(X, 1), j in 1:size(X, 2)];
plt_etas = contour(x, y, Z, xlabel="eta[1]", ylabel="eta[2]", title="Etas prior", label="", colorbar=nothing);

plot(plt_u0, plt_etas, layout=(2,1), size = (800, 600))

priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    "etas_prior" => etas_prior
    );

@model function model_u0(data, times, weight, age, I, priors)
    u01 ~ priors["u01_prior"]
    u02 ~ priors["u02_prior"]
    etas ~ priors["etas_prior"] 

    u0_ = [u01, u02]

    predicted = predict_pk_bjorkman(weight, age, I, times; save_idxs=[1], σ=0, etas=etas, u0=u0_, tspan=(-0.1, last_time + 10));

    data ~ MultivariateNormal(vec(predicted), sigma)

    return nothing
end

# Build model
model = model_u0(data, times, weight, age, I, priors);

# Sample from model
chain = sample(model, NUTS(0.65), MCMCSerial(), 2000, 3; progress=true);
plot(chain)


# Sample n u0s
n = 100
posterior_samples = sample(chain[[:u01, :u02, Symbol("etas[1]"), Symbol("etas[2]")]], n, replace=false);
saveat = collect(0:0.1:last_time + 10);

# Plot solutions for all the sampled parameters
plt = plot(title="n =  $n")
for p in eachrow(Array(posterior_samples))
    sample_u01, sample_u02, sample_eta1, sample_eta2 = p

    # Set initial values
    u0_ = [sample_u01, sample_u02]

    predicted = predict_pk_bjorkman(weight, age, I, saveat; save_idxs=[1], σ=0, etas=[sample_eta1, sample_eta2], u0=u0_, tspan=(-0.1, last_time + 10));

    # Plot predicted pk
    plot!(plt, saveat, predicted, alpha=0.2, color="#BBBBBB", label="");
end
# Plot observed values
scatter!(plt, times, data, color="red", label="Observed values")
display(plt)