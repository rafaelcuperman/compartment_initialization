using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using ArviZ
using Random

include(srcdir("mcmc.jl"));
include(srcdir("metrics.jl"));

save_plots = false;

type_prior = "continuous";
pk_model_data = "mceneny"; # Model that was used to generate the data
pk_model_selection = "mceneny"; # Model that will be used to model the data and make predictions

df = CSV.read(datadir("exp_pro", "variable_times", "$(pk_model_data)_population_1h.csv"), DataFrame);
df.ffm = df.weight*(1-0.3);

if pk_model_selection == "bjorkman"
    include(srcdir("bjorkman.jl"));

    pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

    sigma_additive = 5
    sigma_proportional = 0

elseif pk_model_selection == "mceneny"
    include(srcdir("mceneny.jl"));

    pkmodel(args...; kwargs...) = predict_pk_mceneny(args...; kwargs...);

    sigma_additive = 0
    sigma_proportional = 0.17

elseif pk_model_selection == "simple"
    include(srcdir("simple_pk.jl"));

    pkmodel(args...; kwargs...) = predict_pk_simple(args...; kwargs...);

    sigma_additive = 5
    sigma_proportional = 0

else
    throw(ExceptionError("Unknown pk_model_selection"))
end;

between_dose = 1; #Time between dose for measurments used for MCMC

ids = unique(df.id);

times_correct = [];
probs_correct = [];

loop_ids = ids[1:5];
for id in loop_ids
    println("----------------Loop $id/$(length(loop_ids))----------------")
    df_ = df[df.id .== id, :];
    df_ = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df_);

    metadata = eval(Meta.parse(df_[1,:metadata]));
    time_real = metadata["time"];
    ind, I = individual_from_df(df_);

    # Define priors
    if type_prior == "discrete"
        # Build discrete normal prior based on categorical distributions
        possible_doses = collect(250:250:4000);
        dist = Normal(I[2], 500);
        probs = [pdf(dist, i) for i in possible_doses];
        probs = probs/sum(probs);
        dose_prior = Categorical(probs).*250;
        burnin = 20000
    elseif type_prior == "continuous"  
        dose_prior = Truncated(Normal(I[2], 500), 0, 4000);
        burnin = 1
    else
        throw(ExceptionError("Unknown type_prior"))
    end;

    etas_prior = MultivariateNormal(zeros(2), build_omega_matrix());
    priors = Dict(
        "dose_prior" => dose_prior,
        "etas_prior" => etas_prior
        );

    times = [24, 48, 72];
    chains = [];
    models = [];
    for (ix, t) in enumerate(times)
        # Run MCMC
        println("Running chain $ix/$(length(times)). Using t=$t")
        mcmcmodel = model_dose_etas(pkmodel, ind, I, priors, t; sigma_additive=sigma_additive, sigma_proportional=sigma_proportional);
        if type_prior == "discrete"
            chain = sample(mcmcmodel, MH(), MCMCThreads(), 100000, 3; progress=true);
        else
            chain = sample(mcmcmodel, NUTS(0.65), MCMCThreads(), 2000, 3; progress=true);
        end
        push!(chains, chain)
        push!(models, mcmcmodel)
    end
    println("Models built")

    println("Calculating marginal likelihood with importance sampling...")
    norm_is, plt_is = importance_sampling(chains, models);

    time_pred = times[argmax(norm_is)]
    push!(times_correct, time_pred == time_real ? 1 : 0);
    push!(probs_correct, norm_is[findfirst(x -> x == time_real, times)]);
end

sum(times_correct)/length(times_correct)
sum(probs_correct)/length(probs_correct)
