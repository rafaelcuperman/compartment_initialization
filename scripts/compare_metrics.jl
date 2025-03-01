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

type_prior = "discrete";
pk_model_data = "bjorkman"; # Model that was used to generate the data
pk_model_selection = "bjorkman"; # Model that will be used to model the data and make predictions (bjorkman, mceneny, or simple)

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

df_ = df[df.id .== 5, :]; #5, 19

between_dose = 24; #Time between dose for measurments used for MCMC
df_ = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df_);

metadata = eval(Meta.parse(df_[1,:metadata]))

ind, I = individual_from_df(df_);

scatter(ind.t, ind.y,  xlims=(0,60))

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
plt_dose = plot_priors_dose(priors);
display(plt_dose)

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

# Plot chains
plt = plot(chains[1][burnin:end])
plt = plot(chains[2][burnin:end])
plt = plot(chains[3][burnin:end])


# Plot distribution of all the chains
plt_dose = density(vcat(chains[1][:D].data...), label="t=24", color = "blue", title="Dose", yticks=nothing);
density!(plt_dose, vcat(chains[2][:D].data...), label="t=48", color = "black");
density!(plt_dose, vcat(chains[3][:D].data...), label="t=72",  color = "green");

plt_eta1 = density(vcat(chains[1][Symbol("etas[1]")].data...), label=nothing, color = "blue", title="eta1", yticks=nothing);
density!(plt_eta1, vcat(chains[2][Symbol("etas[1]")].data...), label=nothing, color = "black");
density!(plt_eta1, vcat(chains[3][Symbol("etas[1]")].data...), label=nothing, color = "green");

plt_eta2 = density(vcat(chains[1][Symbol("etas[2]")].data...), label=nothing, color = "blue",  title="eta2", yticks=nothing);
density!(plt_eta2, vcat(chains[2][Symbol("etas[2]")].data...), label=nothing, color = "black");
density!(plt_eta2, vcat(chains[3][Symbol("etas[2]")].data...), label=nothing, color = "green");

plot(plt_dose, plt_eta1, plt_eta2, layout=(3,1)) 


# Plot the distribution of each chain
plt_dose = density(chains[1][:D].data, label="t=24", color = "blue", title="Dose", yticks=nothing);
density!(plt_dose, chains[2][:D].data, label="t=48", color = "black");
density!(plt_dose, chains[3][:D].data, label="t=72",  color = "green");

plt_eta1 = density(chains[1][Symbol("etas[1]")].data, label=nothing, color = "blue", title="eta1", yticks=nothing);
density!(plt_eta1, chains[2][Symbol("etas[1]")].data, label=nothing, color = "black");
density!(plt_eta1, chains[3][Symbol("etas[1]")].data, label=nothing, color = "green");

plt_eta2 = density(chains[1][Symbol("etas[2]")].data, label=nothing, color = "blue",  title="eta2", yticks=nothing);
density!(plt_eta2, chains[2][Symbol("etas[2]")].data, label=nothing, color = "black");
density!(plt_eta2, chains[3][Symbol("etas[2]")].data, label=nothing, color = "green");

# Remove duplicated labels in plot
labels = [x[:label] for x in plt_dose.series_list];

function duplicated_indices(v)
    counts = Dict{eltype(v), Int}()
    duplicates = []
    for i in 1:length(v)
        counts[v[i]] = get(counts, v[i], 0) + 1
        if counts[v[i]] > 1
            push!(duplicates, i)
        end
    end
    return duplicates
end;
indices = duplicated_indices(labels);

for i in indices
    plt_dose.series_list[i][:label] = ""
end;

plot(plt_dose, plt_eta1, plt_eta2, layout=(3,1)) 


#####################################################################
#         Option 1: p(x|θ)p(θ) with θ = mean of posterior           #
#####################################################################
println("Calculating joint probability with posterior mean...")
norm_mean, plt_mean = joint_posterior_mean(chains, models);

#####################################################################
#                       Option 2: simple MCMC                       #
#####################################################################
println("Calculating marginal likelihood with prior distributions...")
norm_ml, plt_ml = marginal_likelihood_approx(chains, models);

#####################################################################
#                   Option 3: importance sampling                   #
#####################################################################
println("Calculating marginal likelihood with importance sampling...")
norm_is, plt_is = importance_sampling(chains, models);

#####################################################################
#                   Option 4 & 5: ELPD LOO & WAIC                   #
#####################################################################
println("Calculating LOO and WAIC...")
norm_loos, norm_waics, plt_loo, plt_waic = loo_waic(chains, models);

#####################################################################
#         Option 6: Marginal Likelihood sampled from posterior      #
#####################################################################
println("Calculating marginal likelihood with posterior samples...")
norm_post, plt_post = ml_post(chains, models);

#####################################################################
#                   Plot all the results
#####################################################################

println("Real time: $(metadata["time"])")
display(plot(plt_mean, plt_post, plt_ml, plt_is, plt_loo, plt_waic, layout=(2,3), size=(800,500)))

metric_names = ["Mean of posterior", "Sampled from posterior", "ML approximation", "Importance sampling", "LOO", "WAIC"];
println(DataFrame(hcat(metric_names, round.(vcat(norm_mean', norm_post', norm_ml', norm_is', norm_loos', norm_waics'), digits=3)), ["Metric", "24h", "48h", "72h"]))


#plot(plt1, plt2, plt4, plt5, layout=(2,2), size=(800,500))
#plot(plt1, plt2, plt5, layout=(2,2), size=(800,500))