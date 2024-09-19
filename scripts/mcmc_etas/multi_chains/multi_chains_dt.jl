using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots

include(srcdir("mcmc.jl"));

save_plots = false

type_prior = "discrete";
pk_model_selection = "bjorkman";

if pk_model_selection == "bjorkman"
    include(srcdir("bjorkman.jl"));

    df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);

    pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

    sigma_additive = 5
    sigma_proportional = 0

else
    include(srcdir("mceneny.jl"));

    df = CSV.read(datadir("exp_pro", "variable_times", "mceneny_population_1h.csv"), DataFrame);
    df.ffm = df.weight*(1-0.3);

    pkmodel(args...; kwargs...) = predict_pk_mceneny(args...; kwargs...);

    sigma_additive = 0
    sigma_proportional = 0.17

end

df_ = df[df.id .== 3, :]; 

between_dose = 1; #Time between dose for measurments used for MCMC
df_ = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), df_);

metadata = eval(Meta.parse(df_[1,:metadata]))

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
else    
    dose_prior = Truncated(Normal(I[2], 500), 0, 4000);
    burnin = 1
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
println("Done")

# Plot chains

plt = plot(chains[1][burnin:end])
plt = plot(chains[2][burnin:end])
plt = plot(chains[3][burnin:end])

#####################################################################
#         Option 1: p(x|θ)p(θ) with θ = mean of posterior           #
#####################################################################

logprobs = [];
pred_doses = [];
pred_etas = [];
for i in 1:length(chains)
    pred_dose = mean(chains[i][:D]);
    pred_eta = [mean(chains[i][Symbol("etas[1]")]), mean(chains[i][Symbol("etas[2]")])];

    pred_dose = round(pred_dose/250)*250;
    logprob = loglikelihood(models[i], (D=pred_dose, etas=pred_eta)) + logprior(models[i], (D=pred_dose, etas=pred_eta))
    #println(logjoint(models[i], (D=pred_dose, etas=pred_etas)))
    push!(logprobs, logprob)
    push!(pred_doses, pred_dose)
    push!(pred_etas, pred_eta)
end

println(metadata)
println(pred_dose)
prob_models = exp.(logprobs)./sum(exp.(logprobs));
bar(times, prob_models, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="Mean of posterior")


#####################################################################
#                   Option 2: simple MCMC
# Simple MCMC to approach the marginal likelihood
# https://leimao.github.io/blog/Marginal-Likelihood-Estimation/
# The marginal likelihood is defined as P(x) = ∫p(x|θ)p(θ)dθ
#
# This is equivalent to P(x) = E[P(x|θ)] with θ~p(θ)
# This can be approximated with P(X) = 1/n * ∑p(x|θᵢ), θᵢ~p(θ)
# With n the number of samples taken for θ. The larger the better
#
# This approach has problems:  https://www.youtube.com/watch?v=1HXAnoalc8Q
# "the variance of the estimator can be very high if the dimension of the conditional variable is high and the number of samples is small."
#####################################################################

mls= [];
for i in 1:length(chains)
    samples_priors = sample(models[i], Prior(), 10000);
    samples_priors = hcat(samples_priors[:D].data, samples_priors[Symbol("etas[1]")].data, samples_priors[Symbol("etas[2]")].data);
    samples_priors = DataFrame(samples_priors, ["D", "eta1", "eta2"]);

    ll = 0;
    #ml = [];
    for (ix, row) in enumerate(eachrow(samples_priors))
        ll += exp(loglikelihood(models[i], (D = row["D"], etas = [row["eta1"], row["eta2"]])))
        #push!(ml, ll/ix)
    end
    #println(ml[end])
    ml = ll/ size(samples_priors, 1)
    push!(mls, ml)
end
norm_mls = mls/sum(mls);
bar(times, norm_mls, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="Simple MCMC")


#####################################################################
#                   Option 3: importance sampling
# https://leimao.github.io/blog/Marginal-Likelihood-Estimation/
# The marginal likelihood is defined as P(x) = ∫p(x|θ)p(θ)dθ
# We can also write it as P(x) = ∫p(x|θ)p(θ)q(θ)/q(θ)dθ with q(θ) a proposal distribution
#
# This is equivalent to P(x) = E[p(x|θ)p(θ)/q(θ)] with θ~q(θ)
# But p(x|θ)p(θ) = p(x,θ), which is the joint distribution. 
# So, P(x) = E[p(x,θ)/q(θ)] with θ~q(θ)
# 
# This can be approximated with P(X) = 1/n * ∑p(x,θᵢ)/q(θᵢ), θᵢ~q(θ)
# With n the number of samples taken for θ. The larger the better
#
# When the q(θ) is the same as the posterior distribution (p(θ|x)), 
# the variance of the estimator is minimized and is equal to zero.
#
# Then, lets use q(θ) = p(θ|x)
#####################################################################

mls_is = [];
for i in 1:length(chains)
    samples_posterior = hcat(vcat(chains[i][burnin:end][:D].data...), vcat(chains[i][burnin:end][Symbol("etas[1]")].data...), vcat(chains[i][burnin:end][Symbol("etas[2]")].data...));
    samples_posterior = DataFrame(samples_posterior, ["D", "eta1", "eta2"]);

    # Build normal distributions based on sample mean and stdev
    posterior_dose = Normal(mean(samples_posterior.D), std(samples_posterior.D));
    posterior_eta1 = Normal(mean(samples_posterior.eta1), std(samples_posterior.eta1));
    posterior_eta2 = Normal(mean(samples_posterior.eta2), std(samples_posterior.eta2));

    x = 0:1:4000;
    plt_dose = plot(x, pdf.(posterior_dose, x), title="Dose posterior", label="", yticks=nothing);
    x = -1:0.01:1;
    plt_eta1 = plot(x, pdf.(posterior_eta1, x), title="Eta1 posterior", label="", yticks=nothing);
    plt_eta2 = plot(x, pdf.(posterior_eta2, x), title="Eta2 posterior", label="", yticks=nothing);
    plot(plt_dose, plt_eta1, plt_eta2, layout=(3,1))

    samples_posterior = last(samples_posterior, 10000);
    ratio = 0;
    #ml = [];
    for (ix, row) in enumerate(eachrow(samples_posterior))
        ljoint = logjoint(models[i], (D = row["D"], etas = [row["eta1"], row["eta2"]]))
        proposal = logpdf(posterior_dose, row["D"]) + logpdf(posterior_eta1, row["eta1"]) + logpdf(posterior_eta2, row["eta2"])
        ratio += exp(ljoint - proposal)
    end
    #println(ml[end])
    ml = ratio/ size(samples_posterior, 1)
    push!(mls_is, ml)
end
norm_mls_is = mls_is/sum(mls_is);
bar(times, norm_mls_is, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="Importance sampling")