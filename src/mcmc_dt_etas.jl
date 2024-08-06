using DrWatson
@quickactivate "compartment-initialization"

using Turing
using DeepCompartmentModels
using Plots

function plot_priors_dt(priors)
    dose_prior = priors["dose_prior"]
    x = range(0, stop=4000, length=1000);
    plt_dose = plot(x, pdf.(dose_prior, x), title="Dose prior", label="", yticks=nothing);

    time_prior = priors["time_prior"]
    x = range(0, stop=40, length=1000);
    plt_time = plot(x, pdf.(time_prior, x), title="Time prior",label="", yticks=nothing);

    etas_prior = priors["etas_prior"]
    x = range(-3, stop=3, length=1000);
    y = range(-3, stop=3, length=1000);
    X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y];
    Z = [pdf(etas_prior, [X[i, j], Y[i, j]]) for i in 1:size(X, 1), j in 1:size(X, 2)];
    plt_etas = contour(x, y, Z, xlabel="eta[1]", ylabel="eta[2]", title="Etas prior", label="", colorbar=nothing);

    display(plot(plt_dose, plt_time, plt_etas, layout=(3,1), size = (800, 600)))
end

function run_chain(pkmodel::Function, ind::BasicIndividual, I::AbstractMatrix, priors::Dict, args...; algo=NUTS(0.65), iters::Int=2000, chains::Int=3, sigma=5, kwargs...)
    @model function model_dt(pkmodel, ind, I, priors, args...; kwargs...)
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
    
        predicted = pkmodel(ind, I_, ind.t .+ t_, args...; save_idxs=[1], σ=0, etas=etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ t_), kwargs...)
    
        ind.y ~ MultivariateNormal(vec(predicted), sigma)
    
        return nothing
    end

    # Build model
    model = model_dt(pkmodel, ind, I, priors);

    # Sample from model
    chain = sample(model, algo, MCMCSerial(), iters, chains; progress=true);
    return chain
end

# Sample n items from the posterior and simulate the curves
function sample_posterior(chain, ind::BasicIndividual, I::AbstractMatrix; n::Int=100)
    # Sample n u0s
    posterior_samples = sample(chain[[:D, :t, Symbol("etas[1]"), Symbol("etas[2]")]], n, replace=false);
    saveat = collect(0:0.1:72);
    
    list_predicted = []
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
    
        predicted = pkmodel(ind, I_, saveat; save_idxs=[1], σ=0, etas=[sample_eta1, sample_eta2], u0=zeros(2), tspan=(-0.1, 72))
        push!(list_predicted, predicted)
    
        # Plot predicted pk centered in t0
        plot!(plt2, saveat .- sample_t, predicted, alpha=0.5, color="#BBBBBB", label="");
    
        # Plot predicted pk restarting t0 as predicted time (observed times)
        start_observed_values = findfirst(x -> x >= sample_t, saveat);
        plot!(plt, saveat[start_observed_values:end] .- sample_t, predicted[start_observed_values:end], alpha=0.2, color="#BBBBBB", label="")
    
    end
    
    # Plot observed values centered at t0
    scatter!(plt2, ind.t, ind.y, color="red", label="Observed values")
    
    # Plot observed values with restarted t0
    scatter!(plt, ind.t, ind.y, color="red", label="Observed values")

    return list_predicted, saveat, plt, plt2
end