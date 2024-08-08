using DrWatson
@quickactivate "compartment-initialization"

using Turing
using DataFrames
using CSV
using Plots
using StatsPlots
using Printf
using GLM

include(srcdir("bjorkman.jl"));

sigma = 5
boolean_etas = "n"

# Read data
df = CSV.read(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), DataFrame);

ind, I = individual_from_df(df);

# Define priors
u0_prior = Truncated(Normal(20, 20), 0, 60);
priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    );


# Plot priors
x = range(u0_prior.lower - 20, stop=u0_prior.upper + 20, length=1000);
plt = plot(x, pdf.(u0_prior, x), title="U0 prior", label="", yticks=nothing);
display(plt)

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

@model function model_u0(pkmodel, ind, I, priors, args...; kwargs...)
    u01 ~ priors["u01_prior"]
    u02 ~ priors["u02_prior"]

    u0_ = [u01, u02]

    predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=zeros(2), u0=u0_, tspan=(-0.1, ind.t[end] + 10), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma)

    return nothing
end

# Build model
model = model_u0(pkmodel, ind, I, priors);

# Sample from model
chain = sample(model, NUTS(0.65), MCMCSerial(), 2000, 3; progress=true);
plot(chain)


# Sample n u0s
n = 100;
posterior_samples = sample(chain[[:u01, :u02]], n, replace=false);
saveat = collect(0:0.1:ind.t[end] + 10);

# Plot solutions for all the sampled parameters
plt = plot(title="n =  $n")
for p in eachrow(Array(posterior_samples))
    sample_u01, sample_u02 = p

    # Set initial values
    u0_ = [sample_u01, sample_u02]

    predicted = pkmodel(ind, I, saveat; save_idxs=[1], σ=0, etas=zeros(2), u0=u0_, tspan=(-0.1, ind.t[end] + 10))

    # Plot predicted pk
    plot!(plt, saveat, predicted, alpha=0.2, color="#BBBBBB", label="");
end
# Plot observed values
scatter!(plt, ind.t, ind.y, color="red", label="Observed values")
display(plt)

##############################################################################
################################# Regression #################################
##############################################################################

# Scatter plot u01 vs u02 of chains
chain_u01 = reshape(chain[:u01].data,:);
chain_u02 = reshape(chain[:u02].data,:);
correlation = cor(chain_u01, chain_u02);
correlation = @sprintf("%.2f", correlation)
plt = scatter(chain_u01, chain_u02, label="", xlabel="u0[1]", ylabel="u0[2]", alpha=0.5, title="Correlation=$correlation")

# Fit a linear regression model
df = DataFrame(chain_u01 = chain_u01, chain_u02 = chain_u02);
lr = lm(@formula(chain_u02 ~ chain_u01), df);
coefs = coef(lr);
d = collect(extrema(chain_u01));
plot!(plt, d, coefs[1] .+ coefs[2] .* d, color="black", linewidth=3, linestyle=:dash, label="Regression line")

# Build u0s based on linear regression
n=100
chain_u01_lr = sample(chain_u01, n, replace=false);
chain_u02_lr = coefs[1] .+ coefs[2] .* chain_u01_lr;
chain_lr = hcat(chain_u01_lr, chain_u02_lr);

# Plot solutions for u0s built with the linear regression
plt = plot(title="n =  $n")
for p in eachrow(Array(chain_lr))
    sample_u01, sample_u02 = p

    # Set initial values
    u0_ = [sample_u01, sample_u02]

    predicted = pkmodel(ind, I, saveat; save_idxs=[1], σ=0, etas=zeros(2), u0=u0_, tspan=(-0.1, ind.t[end] + 10))

    # Plot predicted pk
    plot!(plt, saveat, predicted, alpha=0.2, color="#BBBBBB", label="");
end

# Plot observed values
scatter!(plt, ind.t, ind.y, color="red", label="Observed values")
display(plt)