using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_u0_etas.jl"));
include(srcdir("bjorkman.jl"));

# Boolean to control if plots are saved
save_plots = false;

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df_ = df[df.id .== 1, :];  #19, 5, 1

between_dose = 1; #Time between dose for measurments used for MCMC
df_ = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df_);

metadata = eval(Meta.parse(df_[1,:metadata]))

ind, I = individual_from_df(df_);

# Define priors
u0_prior = Truncated(Exponential(10), 0, 60);
etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
priors = Dict(
    "u01_prior" => u0_prior,
    "u02_prior" => u0_prior,
    "etas_prior" => etas_prior
    );

# Plot priors
plt = plot_priors_u0(priors);
#display(plt)

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

# Run MCMC
chain = run_chain(pkmodel, ind, I, priors; algo=NUTS(0.65), iters=2000, chains=3, sigma=5);
plt = plot(chain)
#save_plots && savefig(plt, plotsdir("chain_multi.png"))

# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;
mode_u01 = mode(round.(chain[:u01].data./round_u0s).*round_u0s);
mode_u02 = mode(round.(chain[:u02].data./round_u0s).*round_u0s);
mode_eta1 = mode(round.(chain[Symbol("etas[1]")].data./round_etas).*round_etas);
mode_eta2 = mode(round.(chain[Symbol("etas[2]")].data./round_etas).*round_etas);
pred_etas = [mode_eta1, mode_eta2];

# Get real values of etas and u0s
real_etas = round.(metadata["etas"]./round_etas).*round_etas;
real_u0s = round.(metadata["u0s"]./round_u0s).*round_u0s;
real_dose = metadata["dose"];
real_time = metadata["time"];

println("Real u0s: $(real_u0s). Pred u0s: $([mode_u01, mode_u02])")
println("Real etas: $(real_etas). Pred etas: $(pred_etas)")


########## Predict dose and time based on predicted etas ###########

#dist = Truncated(MixtureModel(map(u -> Normal(u, 3), 12:12:72)), 0, 96);
#x = range(0, stop=120, length=1000);
#plot(x, pdf.(dist, x), yticks=nothing)

#dist = Truncated(Normal(I[2], 100), 0, 5000);
#x = range(0, stop=5000, length=1000);
#plot(x, pdf.(dist, x), yticks=nothing)

@model function model_dt(pkmodel, ind, I, etas, args...; kwargs...)
    #t ~ Truncated(MixtureModel(map(u -> Normal(u, 2), 24:24:72)), 0, 96);
    #t ~ Truncated(Normal(48,48), 0, 96);
    t ~ Categorical(ones(3)/3).*24;

    D ~ Truncated(Normal(I[2], 100), 0, 5000);
    #D = I[2]

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

    ind.y ~ MultivariateNormal(vec(predicted), 5)

    return nothing
end

# Build model
model2 = model_dt(pkmodel, ind, I, pred_etas);

# Sample from model
#initial_params = FillArrays.fill([(weights = [0.2, 0.5, 0.3], t = 72, etas = [0,0])], 3)
#chain2 = sample(model2, NUTS(0.65), MCMCSerial(), 1000, 3; progress=true);
chain2 = sample(model2, MH(), MCMCSerial(), 10000, 3; progress=true);
plot(chain2[5000:end])

round_time = 24;
round_dose = 250;
pred_time = mode(round.(chain2[:t].data./round_time).*round_time);
pred_time = mode(chain2[:t]);
pred_dose = mode(round.(chain2[:D].data./round_dose).*round_dose);

println("Real time: $(real_time). Pred time: $(pred_time)")
println("Real dose: $(real_dose). Pred dose: $(pred_dose)")



###### Max loglikelihood ######
dist = MultivariateNormal(ind.y, 5);
ll = Dict()
for time in [24, 48, 72]
    # Regenerate initial dose
    I_ = copy(I);
    I_ = vcat([0. 0. 0. 0.], I_);  
    
    # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
    I_[2:end, 1] = I_[2:end, 1] .+ time;
    I_[1,2] = real_dose;
    I_[1,3] = real_dose*60;
    I_[1,4] = 1/60;
    
    # Get predictions
    predicted = pkmodel(ind, I_, ind.t .+ time; save_idxs=[1], σ=0, etas=pred_etas, u0=zeros(2), tspan=(-0.1, ind.t[end] .+ time));
    
    # Get likelihood of predictions with respect to the posterior of the observed values
    ll[time] = log(pdf(dist, vec(predicted)))
end
println(ll)
plot(chain2)