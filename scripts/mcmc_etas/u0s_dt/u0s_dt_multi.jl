using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc_u0_etas.jl"));
include(srcdir("bjorkman.jl"));


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

# Boolean to control if plots are saved
save_plots = false;

# Rounding parameters for u0s and etas
round_u0s = 1;
round_etas = 0.1;

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = filter(row -> row.id <= 30, df); # First 30 patients

between_dose = 1; #Time between dose for measurments used for MCMC
df = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df);

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

real_etas = [];
real_u0s = [];
real_times = [];
real_doses = [];

pred_etas = [];
pred_u0s1 = [];

pred_times = [];
pred_doses = [];
pred_u0s2 = [];

for (ix, i) in enumerate(unique(df.id))
    #if ix == 6
    #    break
    #end    

    println("$ix/$(length(unique(df.id)))")

    # Filter patient i
    df_ = filter(row -> row.id == i, df)
    ind, I = individual_from_df(df_);

    # Filter observations that will be used for MCMC. The rest are used only for evaluation
    df_use = filter(row -> (row.mdv == 1) .| (row.time ∈ 1:between_dose:ind.t[end]), df_);
    ind_use, I_use = individual_from_df(df_use);

    # Run MCMC
    chain = run_chain(pkmodel, ind_use, I_use, priors; algo=NUTS(0.65), iters=2000, chains=1, sigma=5);

    # Get predicted etas and u0s
    mode_u011 = mode(round.(chain[:u01].data./round_u0s).*round_u0s);
    mode_u021 = mode(round.(chain[:u02].data./round_u0s).*round_u0s);
    mode_eta1 = mode(round.(chain[Symbol("etas[1]")].data./round_etas).*round_etas);
    mode_eta2 = mode(round.(chain[Symbol("etas[2]")].data./round_etas).*round_etas);

    # Get real values
    metadata = eval(Meta.parse(df_[1,:metadata]))
    real_eta = round.(metadata["etas"]./round_etas).*round_etas;
    real_u0 = round.(metadata["u0s"]./round_u0s).*round_u0s;
    real_dose = metadata["dose"];
    real_time = metadata["time"];

    ########## Predict dose and time based on predicted etas ###########
    model2 = model_dt(pkmodel, ind_use, I_use, [mode_eta1, mode_eta2]);
    chain2 = sample(model2, MH(), MCMCSerial(), 10000, 3; progress=true);

    pred_time = mode(round.(chain2[5000:end][:t].data./round_time).*round_time);
    pred_dose = mode(round.(chain2[5000:end][:D].data./round_dose).*round_dose);

    push!(real_etas, real_eta) 
    push!(real_u0s, real_u0);
    push!(real_times, real_time);
    push!(real_doses, real_dose);

    push!(pred_etas, [mode_eta1, mode_eta2]);
    push!(pred_u0s1, [mode_u011, mode_u021]);

    push!(pred_times, pred_time)
    push!(pred_doses, pred_dose)

    ######### Forward to u0 based on predicted dose and time ##########
    # Get predictions
    predicted = []
    posterior_samples = sample(chain2[5000:end][[:t,:D]], 100, replace=false)
    for p in eachrow(Array(posterior_samples))
        t, D = p

        # Regenerate initial dose
        I_ = copy(I_use);
        I_ = vcat([0. 0. 0. 0.], I_);  

        # Shift all the dosing times by t_ (predicted time of second dose) except the initial dose that is at t=0
        I_[2:end, 1] = I_[2:end, 1] .+ t;
        I_[1,2] = D;
        I_[1,3] = D*60;
        I_[1,4] = 1/60;
        
        push!(predicted, pkmodel(ind_use, I_, [t]; save_idxs=[1, 2], σ=5, etas=[mode_eta1, mode_eta2], u0=zeros(2), tspan=(-0.1, ind.t[end] .+ pred_time)))
    end
    push!(pred_u0s2, round.(mean(vcat(predicted...), dims=1))./round_u0s).*round_u0s;

end

print(real_doses)
println(pred_doses)

print(real_times)
println(pred_times)

print(real_etas)
println(pred_etas)

print(real_u0s)
print(pred_u0s1)
println(pred_u0s2)


plt = plot(real_doses)
plot!(plt, pred_doses)

plt = plot(real_times)
plot!(plt, pred_times)

plt = plot(hcat(real_u0s...)[1,:])
plot!(plt, hcat(pred_u0s1...)[1,:])
plot!(plt, vcat(pred_u0s2...)[:,1])

plt = plot(hcat(real_u0s...)[2,:])
plot!(plt, hcat(pred_u0s1...)[2,:])
plot!(plt, vcat(pred_u0s2...)[:,2])
