using DrWatson
@quickactivate "compartment-initialization"

using Turing
using DataFrames
using CSV
using Plots
using StatsPlots
using Printf
using GLM
using KernelDensity

include(srcdir("bjorkman.jl"));

sigma = 5
boolean_etas = "n";

# Read data
df = CSV.read(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), DataFrame);

data = Float64.(df[2:end, :dv]);
times = Float64.(df[2:end, :time]);

# Reconstruct dosing matrix
I = reshape(Float64.(collect(df[1, [:time, :amt, :rate, :duration]])),1,4);

# Define priors
dose_prior = Truncated(Normal(1750, 1000), 1000, 3000);
time_prior = Truncated(Normal(12, 10), 6, 36);

priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    );

    
@model function model_dt(data, times, weight, age, I, priors)
    D ~ priors["dose_prior"]
    t ~ priors["time_prior"]

    # The dosing callback function requires integer values
    D_ = round(D)
    t_ = round(t)

    # Regenerate initial dose
    I_ = copy(I)
    I_ = vcat([0. 0. 0. 0.], I_)  

    # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ t_
    I_[1,2] = D_
    I_[1,3] = D_*60
    I_[1,4] = 1/60


    predicted = predict_pk_bjorkman(weight, age, I_, times .+ t_, save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72));

    data ~ MultivariateNormal(vec(predicted), sigma)

    return nothing
end

# Build model
model = model_dt(data, times, weight, age, I, priors);

# Sample from model
chain = sample(model, NUTS(0.65), MCMCSerial(), 1000, 3; progress=true);
plot(chain)


# Sample n Doses and times
n = 100
posterior_samples = sample(chain[[:D, :t]], n, replace=false);

# Plot solutions for all the sampled parameters
plt = plot(title="n =  $n")
for p in eachrow(Array(posterior_samples))
    sample_D, sample_t = p

    # Regenerate initial dose
    I_ = copy(I)
    I_ = vcat([0. 0. 0. 0.], I_)  

    # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ sample_t
    I_[1,2] = sample_D
    I_[1,3] = sample_D*60
    I_[1,4] = 1/60

    predicted = predict_pk_bjorkman(weight, age, I_, saveat; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72));
    
    # Plot predicted pk restarting t0 as predicted time (observed times)
    start_observed_values = findfirst(x -> x >= sample_t, saveat);
    plot!(plt, saveat[start_observed_values:end] .- sample_t, predicted[start_observed_values:end], alpha=0.2, color="#BBBBBB", label="")
end

# Plot observed values with observed_times
scatter!(plt, times, data, color="red", label="Observed values")

display(plt)

##############################################################################
################################# Regression #################################
##############################################################################

# Scatter plot D vs t of chains
chain_D = reshape(chain[:D].data,:);
chain_t = reshape(chain[:t].data,:);
correlation = cor(chain_D, chain_t);
correlation = @sprintf("%.2f", correlation)
plt = scatter(chain_D, chain_t, label="", xlabel="Dose", ylabel="Time", alpha=0.5, title="Correlation=$correlation")

# Fit a linear regression model
df = DataFrame(chain_D = chain_D, chain_t = chain_t);
#df = df[(df.chain_D .>= 1000) .& (df.chain_D .<= 3000), :] # Filter so that Dose is larger than 1000 and smaller than 3000
lr = lm(@formula(chain_t ~ chain_D), df);
coefs = coef(lr);
d = collect(extrema(chain_D));
plot!(plt, d, coefs[1] .+ coefs[2] .* d, color="black", linewidth=3, linestyle=:dash, label="Regression line")

# Build pairs (Dose, time) based on linear regression
n=100
chain_D_lr = sample(chain_D, n, replace=false);
chain_t_lr = coefs[1] .+ coefs[2] .* chain_D_lr;
chain_lr = hcat(chain_D_lr, chain_t_lr);

# Plot solutions for (dose, time) pairs built with the linear regression
plt = plot(title="n =  $n")
for p in eachrow(Array(chain_lr))
    sample_D, sample_t = p

    # Regenerate initial dose
    I_ = copy(I)
    I_ = vcat([0. 0. 0. 0.], I_)  

    # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ sample_t
    I_[1,2] = sample_D
    I_[1,3] = sample_D*60
    I_[1,4] = 1/60

    predicted = predict_pk_bjorkman(weight, age, I_, saveat; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72));
    
    # Plot predicted pk restarting t0 as predicted time (observed times)
    start_observed_values = findfirst(x -> x >= sample_t, saveat);
    plot!(plt, saveat[start_observed_values:end] .- sample_t, predicted[start_observed_values:end], alpha=0.2, color="#BBBBBB", label="")
end
# Plot observed values with observed_times
scatter!(plt, times, data, color="red", label="Observed values")

display(plt)

##############################################################################
#################################### U0s #####################################
##############################################################################

# Sample n Doses and times
n = 100
posterior_samples = sample(chain[[:D, :t]], n, replace=false);

# Get u0s from sampled dose and time. u0 are the concentration values just before the new dose is given
u0s = zeros(n,2);
for (ix, p) in enumerate(eachrow(Array(posterior_samples)))
    sample_D, sample_t = p

    # Regenerate initial dose
    I_ = copy(I)
    I_ = vcat([0. 0. 0. 0.], I_)  

    # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ sample_t
    I_[1,2] = sample_D
    I_[1,3] = sample_D*60
    I_[1,4] = 1/60

    # Get PK predictions
    predicted = predict_pk_bjorkman(weight, age, I_, saveat; save_idxs=[1,2], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72));

    # Take the initial values as the values one step before the new dose is given
    t0 = findfirst(x -> x >= sample_t, saveat);
    u0s[ix,:] = predicted[t0-1,:]
end

# Plot univariate distributions of u0s
pltu01 = histogram(u0s[:,1], bins=50, label="", xlabel="u0[1]", normalize=true);
kde_u01 = kde(u0s[:,1]);
plot!(pltu01, kde_u01.x, kde_u01.density, label="", color="black",  linewidth=2);

pltu02 = histogram(u0s[:,2], bins=50, label="", xlabel="u0[2]", normalize=true);
kde_u02 = kde(u0s[:,2]);
plot!(pltu02, kde_u02.x, kde_u02.density, label="", color="black",  linewidth=2);

plot(pltu01, pltu02, layout=(2,1), size = (800, 600))


# Surface plot of bivariate distribution
plt = plot(xaxis="u0[1]", yaxis="u0[2]")
scatter!(plt, u0s[:,1], u0s[:,2], label="",  color="black", markersize=1)
kde_u0 = kde(u0s)
contour!(plt, kde_u0, levels=10)


# Plot reconstructed PK starting at the second dose with sampled u0s
n_u0s = 100
sample_u0s = sample(eachrow(u0s), n_u0s, replace = false);
plt = plot(title="n =  $n_u0s")
for i in sample_u0s

    # Get PK predictions
    predicted = predict_pk_bjorkman(weight, age, I, saveat; save_idxs=[1], σ=0, etas=zeros(2), u0=i, tspan=(-0.1, 72));

    # Plot reconstructed model
    plot!(plt, saveat, predicted, alpha=0.2, color="#BBBBBB", label="")

end

# Plot observed values with observed_times
scatter!(plt, times, data, color="red", label="Observed values")

display(plt)