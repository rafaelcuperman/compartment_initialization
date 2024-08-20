using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using Turing

include(srcdir("bjorkman.jl"));

# Boolean to control if plots are saved
save_plots = false;

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = filter(row -> row.id <= 30, df); # First 30 patients

# Run MCMC
function run_chain_etas(pkmodel::Function, ind::BasicIndividual, I::AbstractMatrix, priors::Dict, dose, time, args...; algo=NUTS(0.65), iters::Int=2000, chains::Int=3, sigma=5, kwargs...)
    @model function model_dt(pkmodel, ind, I, priors, dose, time, args...; kwargs...)
        etas ~ priors["etas_prior"] 
    
        # Regenerate initial dose
        I_ = copy(I)
        I_ = vcat([0. 0. 0. 0.], I_)  
    
        # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
        I_[2:end, 1] = I_[2:end, 1] .+ time
        I_[1,2] = dose
        I_[1,3] = dose*60
        I_[1,4] = 1/60
    
        predicted = pkmodel(ind, I_, ind.t .+ time, args...; save_idxs=[1], σ=0, etas=etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ time), kwargs...)
    
        ind.y ~ MultivariateNormal(vec(predicted), sigma)
    
        return nothing
    end

    # Build model
    model = model_dt(pkmodel, ind, I, priors, dose, time);

    # Sample from model
    chain = sample(model, algo, MCMCSerial(), iters, chains; progress=true);
    return chain
end

# Define priors
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "etas_prior" => etas_prior
    );

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);
sigma = 5;

pred_times = [];
real_times = [];
for (ix, i) in enumerate(unique(df.id))
    #if ix == 5
    #    break
    #end 

    println()
    println("$ix/$(length(unique(df.id)))")
    println()

    df_ = df[df.id .== i, :];

    ind, I = individual_from_df(df_);

    dose = Int(I[2])
    # Posterior built with the observed data
    dist = MultivariateNormal(ind.y, 5);
    metadata = eval(Meta.parse(df_[1,:metadata]));
    push!(real_times, metadata["time"]);

    ll = Dict()
    for time in [24, 48, 72]

        chain = run_chain_etas(pkmodel, ind, I, priors, dose, time; algo=NUTS(0.65), iters=2000, chains=1, sigma=sigma);

        # Rounding parameters for etas
        round_etas = 0.1;
        
        # Get parameters and modes
        mode_eta1 = mode(round.(chain[Symbol("etas[1]")].data./round_etas).*round_etas);
        mode_eta2 = mode(round.(chain[Symbol("etas[2]")].data./round_etas).*round_etas);
        
        #println("etas real: $(round.(metadata["etas"]./round_etas).*round_etas)")
        #println("etas predicted: $([mode_eta1, mode_eta2])")
        
        # Regenerate initial dose
        I_ = copy(I);
        I_ = vcat([0. 0. 0. 0.], I_);  
        
        # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
        I_[2:end, 1] = I_[2:end, 1] .+ time;
        I_[1,2] = dose;
        I_[1,3] = dose*60;
        I_[1,4] = 1/60;
        
        # Get predictions
        predicted = pkmodel(ind, I_, ind.t .+ time; save_idxs=[1], σ=0, etas=[mode_eta1, mode_eta2], u0=zeros(2), tspan=(-0.1, ind.t[end] .+ time));
        
        # Get likelihood of predictions with respect to the posterior of the observed values
        ll[time] = log(pdf(dist, vec(predicted)))
    end
    # Get index (time) of max likelihood
    push!(pred_times, findmax(ll)[2])
end

# Print number of correct time predictions
println(round(100*sum(real_times .== pred_times)/length(real_times)))