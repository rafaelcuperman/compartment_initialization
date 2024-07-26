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

# Reconstruct dosing matrix
I = reshape(Float64.(collect(df[1, [:time, :amt, :rate, :duration]])),1,4)

scatter(times, data, color="red", label="Observed values")

# Define priors
dose_prior = Truncated(Normal(1750, 1000), 1000, 3000);
time_prior = Truncated(Normal(12, 10), 6, 36);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());

priors = Dict(
    "dose_prior" => dose_prior,
    "time_prior" => time_prior,
    "etas_prior" => etas_prior
    );

    
@model function model_dt(data, times, weight, age, I, priors)
    D ~ priors["dose_prior"]
    t ~ priors["time_prior"]
    etas ~ priors["etas_prior"] 

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


    predicted = predict_pk_bjorkman(weight, age, I_, times .+ t_, save_idxs=[1], σ=0, etas=etas, u0=zeros(2), tspan=(-0.1, 72));

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
posterior_samples = sample(chain[[:D, :t, Symbol("etas[1]"), Symbol("etas[2]")]], n, replace=false);

# Plot solutions for all the sampled parameters
plt = plot(title="n =  $n")
plt2 = plot(title="n =  $n")
for p in eachrow(Array(posterior_samples))
    sample_D, sample_t, sample_eta1, sample_eta2 = p

    # Regenerate initial dose
    I_ = copy(I)
    I_ = vcat([0. 0. 0. 0.], I_)  

    # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ sample_t
    I_[1,2] = sample_D
    I_[1,3] = sample_D*60
    I_[1,4] = 1/60

    predicted = predict_pk_bjorkman(weight, age, I_, saveat; save_idxs=[1], σ=0, etas=[sample_eta1, sample_eta2], u0=zeros(2), tspan=(-0.1, 72));

    # Plot predicted pk centered in t0
    plot!(plt2, saveat .- sample_t, predicted, alpha=0.5, color="#BBBBBB", label="");

    # Plot predicted pk restarting t0 as predicted time (observed times)
    start_observed_values = findfirst(x -> x >= sample_t, saveat);
    plot!(plt, saveat[start_observed_values:end] .- sample_t, predicted[start_observed_values:end], alpha=0.2, color="#BBBBBB", label="")

end

# Plot observed values with observed_times
scatter!(plt2, times, data, color="red", label="Observed values")
display(plt2)

# Plot observed values with observed_times
scatter!(plt, times, data, color="red", label="Observed values")
display(plt)