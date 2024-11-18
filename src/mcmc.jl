using DrWatson
@quickactivate "compartment-initialization"

using Turing
using DeepCompartmentModels: BasicIndividual, generate_dosing_callback
using Plots

function plot_priors_u0(priors)
    u01_prior = priors["u01_prior"]
    x = range(-10, stop=50, length=1000);
    plt_u01 = plot(x, pdf.(u01_prior, x), title="u01 prior", label="", yticks=nothing);

    u02_prior = priors["u02_prior"]
    x = range(-10, stop=100, length=1000);
    plt_u02 = plot(x, pdf.(u02_prior, x), title="u02 prior", label="", yticks=nothing);

    etas_prior = priors["etas_prior"]
    x = range(-3, stop=3, length=1000);
    y = range(-3, stop=3, length=1000);
    X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y];
    Z = [pdf(etas_prior, [X[i, j], Y[i, j]]) for i in 1:size(X, 1), j in 1:size(X, 2)];
    plt_etas = contour(x, y, Z, xlabel="eta[1]", ylabel="eta[2]", title="Etas prior", label="", colorbar=nothing);

    return plt_u01, plt_u02
end

function plot_priors_etas(priors)
    etas_prior = priors["etas_prior"]
    x = range(-3, stop=3, length=1000);
    y = range(-3, stop=3, length=1000);
    X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y];
    Z = [pdf(etas_prior, [X[i, j], Y[i, j]]) for i in 1:size(X, 1), j in 1:size(X, 2)];
    plt_etas = contour(x, y, Z, xlabel="eta[1]", ylabel="eta[2]", title="Etas prior", label="", colorbar=nothing);

    return plt_etas
end

function plot_priors_dose(priors)
    dose_prior = priors["dose_prior"]
    x = 0:1:4000;
    plt_dose = plot(x, pdf.(dose_prior, x), title="Dose prior", label="", yticks=nothing);

   return plt_dose
end

function plot_priors_time(priors)
    time_prior = priors["time_prior"]
    x = range(0, stop=120, length=1000);
    plt_time = plot(x, pdf.(time_prior, x), title="Time prior",label="", yticks=nothing);

   return plt_time
end

#model = model_u0(pkmodel, ind, I, priors);
function run_chain(mcmc_model; algo=NUTS(0.65), iters::Int=2000, chains::Int=3)
    chain = sample(mcmc_model, algo, MCMCSerial(), iters, chains; progress=true);
    return chain
end

@model function model_u0_etas(pkmodel, ind, I, priors, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
    u01 ~ priors["u01_prior"]
    u02 ~ priors["u02_prior"]
    etas ~ priors["etas_prior"] 

    u0_ = [u01, u02]

    if any(abs.(etas) .> 4)
        Turing.@addlogprob! -Inf
        return
    end

    predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=etas, u0=u0_, tspan=(-0.1, ind.t[end] + 10), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end

@model function model_u0(pkmodel, ind, I, priors, etas, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
    u01 ~ priors["u01_prior"]
    u02 ~ priors["u02_prior"]

    u0_ = [u01, u02]

    predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=etas, u0=u0_, tspan=(-0.1, ind.t[end] + 10), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end

@model function model_dt_etas(pkmodel, ind, I, priors, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
    D ~ priors["dose_prior"]
    t ~ priors["time_prior"]
    etas ~ priors["etas_prior"] 

    if any(abs.(etas) .> 4)
        Turing.@addlogprob! -Inf
        return
    end

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

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end


@model function model_dt(pkmodel, ind, I, priors, etas, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
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

    predicted = pkmodel(ind, I_, ind.t .+ t_, args...; save_idxs=[1], σ=0, etas=etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ t_), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end

@model function model_etas(pkmodel, ind, I, priors, u0s, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
    etas ~ priors["etas_prior"] 

    if any(abs.(etas) .> 4)
        Turing.@addlogprob! -Inf
        return
    end

    predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=etas, u0=u0s, tspan=(-0.1, ind.t[end] + 10), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end

@model function model_dose_etas(pkmodel, ind, I, priors, t, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
    D ~ priors["dose_prior"]
    etas ~ priors["etas_prior"] 

    if any(abs.(etas) .> 4)
        Turing.@addlogprob! -Inf
        return
    end

    # The dosing callback function requires integer values
    D_ = round(D)
    t_ = round(t)

    # Regenerate initial dose
    I_ = copy(I)
    I_ = vcat([0. 0. 0. 0.], I_)  

    # Shift all the dosing times by t_ (time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ t_
    I_[1,2] = D_
    I_[1,3] = D_*60
    I_[1,4] = 1/60

    predicted = pkmodel(ind, I_, ind.t .+ t_, args...; save_idxs=[1], σ=0, etas=etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ t_), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end

@model function model_u01_etas(pkmodel, ind, I, priors, args...; sigma_additive=5, sigma_proportional=0.17, kwargs...)
    u01 ~ priors["u01_prior"]
    etas ~ priors["etas_prior"] 

    u0_ = [u01, 0]

    if any(abs.(etas) .> 4)
        Turing.@addlogprob! -Inf
        return
    end

    predicted = pkmodel(ind, I, ind.t, args...; save_idxs=[1], σ=0, etas=etas, u0=u0_, tspan=(-0.1, ind.t[end] + 10), kwargs...)

    ind.y ~ MultivariateNormal(vec(predicted), sigma_additive .+ vec(predicted).*sigma_proportional)

    return nothing
end

# Sample n items from the posterior and simulate the curves
function sample_posterior_u0_eta(chain, ind::BasicIndividual, I::AbstractMatrix; n::Int=100, saveat=ind.t)
    # Sample n u0s
    posterior_samples = sample(chain[[:u01, :u02, Symbol("etas[1]"), Symbol("etas[2]")]], n, replace=false);

    saveat = saveat isa AbstractVector ? saveat : collect(0:saveat:ind.t[end] + 10);

    list_predicted = []
    ps = []
    # Plot solutions for all the sampled parameters
    plt = plot(title="n =  $n")
    for p in eachrow(Array(posterior_samples))
        sample_u01, sample_u02, sample_eta1, sample_eta2 = p
        push!(ps, p)

        # Set initial values
        u0_ = [sample_u01, sample_u02]

        predicted = pkmodel(ind, I, saveat; save_idxs=[1], σ=0, etas=[sample_eta1, sample_eta2], u0=u0_, tspan=(-0.1, ind.t[end] + 10))
        push!(list_predicted, predicted)

        # Plot predicted pk
        plot!(plt, saveat, predicted, alpha=0.2, color="#BBBBBB", label="");
    end
    # Plot observed values
    scatter!(plt, ind.t, ind.y, color="red", label="Observed values")

    return list_predicted, saveat, ps, plt
end


# Sample n items from the posterior and simulate the curves
function sample_posterior_dt_eta(chain, ind::BasicIndividual, I::AbstractMatrix; n::Int=100, saveat=ind.t, plot_scatter=true)
    # Sample n u0s
    posterior_samples = sample(chain[[:D, :t, Symbol("etas[1]"), Symbol("etas[2]")]], n, replace=false);
    
    saveat = saveat isa AbstractVector ? saveat : collect(0:saveat:120);
    
    list_predicted = []
    ps = []
    # Plot solutions for all the sampled parameters
    plt = plot(title="n =  $n")
    plt2 = plot(title="n =  $n")
    for p in eachrow(Array(posterior_samples))
        sample_D, sample_t, sample_eta1, sample_eta2 = p
        push!(ps,p)
    
        # Regenerate initial dose
        I_ = copy(I)
        I_ = vcat([0. 0. 0. 0.], I_)  
    
        # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
        I_[2:end, 1] = I_[2:end, 1] .+ sample_t
        I_[1,2] = sample_D
        I_[1,3] = sample_D*60
        I_[1,4] = 1/60
    
        predicted = pkmodel(ind, I_, saveat; save_idxs=[1], σ=0, etas=[sample_eta1, sample_eta2], u0=zeros(2), tspan=(-0.1, 120))
        push!(list_predicted, predicted)
    
        # Plot predicted pk centered in t0
        plot!(plt2, saveat .- sample_t, predicted, alpha=0.5, color="#BBBBBB", label="");
    
        # Plot predicted pk restarting t0 as predicted time (observed times)
        start_observed_values = findfirst(x -> x >= sample_t, saveat);
        plot!(plt, saveat[start_observed_values:end] .- sample_t, predicted[start_observed_values:end], alpha=0.2, color="#BBBBBB", label="")
    
    end
    
    if plot_scatter
        # Plot observed values centered at t0
        scatter!(plt2, ind.t, ind.y, color="red", label="Observed values")
        
        # Plot observed values with restarted t0
        scatter!(plt, ind.t, ind.y, color="red", label="Observed values")
    end

    return list_predicted, saveat, ps, plt, plt2
end

# Forward pass with predicted time, dose and etas
function forward_u0(pkmodel, chain, ind::BasicIndividual, I::AbstractMatrix, etas; n::Int=100)
    posterior_samples = sample(chain[[:D, :t]], n, replace=false);

    u01_forward = []
    u02_forward = []
    for p in eachrow(Array(posterior_samples))
        sample_D, sample_t = p
        # Regenerate initial dose
        I_ = copy(I);
        I_ = vcat([0. 0. 0. 0.], I_);

        # Shift all the dosing times by predicted time except the initial dose that is at t=0
        I_[2:end, 1] = I_[2:end, 1] .+ sample_t;
        I_[1,2] = sample_D;
        I_[1,3] = sample_D*60;
        I_[1,4] = 1/60;

        u0_forward = pkmodel(ind, I_, [sample_t]; save_idxs=[1,2], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ sample_t))

        push!(u01_forward, u0_forward[1])
        push!(u02_forward, u0_forward[2])
    end
    return u01_forward, u02_forward
end