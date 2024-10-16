using DrWatson
@quickactivate "compartment-initialization"

#####################################################################
#         Option 1: p(x|θ)p(θ) with θ = mean of posterior           #
#####################################################################
function joint_posterior_mean(chains, models)
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
    prob_models = exp.(logprobs)./sum(exp.(logprobs));
    plt = bar(times, prob_models, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="Mean of posterior")
    return prob_models, plt
end

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

function marginal_likelihood_approx(chains, models)
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
    plt = bar(times, norm_mls, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="ML approximation")
    return norm_mls, plt
end


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
# Then, lets use q(θ) = p(θ|x) approximating p(θ|x) as Normal distributions
# with mean and stdev their respective values from the chains
#####################################################################

function importance_sampling(chains, models)
    mls_is = [];
    for i in 1:length(chains)
        samples_posterior = hcat(vcat(chains[i][burnin:end][:D].data...), vcat(chains[i][burnin:end][Symbol("etas[1]")].data...), vcat(chains[i][burnin:end][Symbol("etas[2]")].data...));
        samples_posterior = DataFrame(samples_posterior, ["D", "eta1", "eta2"]);
    
        # Build normal distributions based on sample mean and stdev
        posterior_dose = Normal(mean(samples_posterior.D), std(samples_posterior.D));
        posterior_eta1 = Normal(mean(samples_posterior.eta1), std(samples_posterior.eta1));
        posterior_eta2 = Normal(mean(samples_posterior.eta2), std(samples_posterior.eta2));
    
        #x = 0:1:4000;
        #plt_dose = plot(x, pdf.(posterior_dose, x), title="Dose posterior", label="", yticks=nothing);
        #x = -1:0.01:1;
        #plt_eta1 = plot(x, pdf.(posterior_eta1, x), title="Eta1 posterior", label="", yticks=nothing);
        #plt_eta2 = plot(x, pdf.(posterior_eta2, x), title="Eta2 posterior", label="", yticks=nothing);
        #plot(plt_dose, plt_eta1, plt_eta2, layout=(3,1))
    
        #samples_posterior = last(samples_posterior, 10000);
        n=10000
        samples_posterior = DataFrame(hcat(rand(posterior_dose, n), rand(posterior_eta1, n), rand(posterior_eta2, n)), ["D", "eta1", "eta2"])
        ratio = 0;
        for (ix, row) in enumerate(eachrow(samples_posterior))
            #ljoint = logjoint(models[i], (D = row["D"], etas = [row["eta1"], row["eta2"]]))
            #proposal = logpdf(posterior_dose, row["D"]) + logpdf(posterior_eta1, row["eta1"]) + logpdf(posterior_eta2, row["eta2"])
            #ratio += exp(ljoint)
            D = round(row["D"]/250)*250;
            llikelihood = loglikelihood(models[i], (D = D, etas = [row["eta1"], row["eta2"]]))
            lprior = logprior(models[i], (D = D, etas = [row["eta1"], row["eta2"]]))
            logproposal = logpdf(posterior_dose, D) + logpdf(posterior_eta1, row["eta1"]) + logpdf(posterior_eta2, row["eta2"])
            ratio += exp(llikelihood + lprior - logproposal)
        end
        #println(ml[end])
        ml = ratio/ size(samples_posterior, 1)
        push!(mls_is, ml)
    end
    norm_mls_is = mls_is/sum(mls_is);
    plt = bar(times, norm_mls_is, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="Importance sampling")

    return norm_mls_is, plt
end

#####################################################################
#                   Option 4 & 5: ELPD LOO & WAIC                   
# https://burtonjosh.github.io/blog/golf-putting-in-turing/
# https://arviz-devs.github.io/ArviZ.jl/stable/api/stats/#PosteriorStats.ModelComparisonResult
#####################################################################

function loo_waic(chains, models)
    loos = [];
    waics = [];
    idatas = Dict();
    for i in 1:length(chains)
        log_likelihood = pointwise_loglikelihoods(models[i], MCMCChains.get_sections(chains[i], :parameters))
        log_likelihood = log_likelihood["ind.y"];
        log_likelihood = reshape(log_likelihood, size(log_likelihood)..., 1);
        push!(loos, loo(log_likelihood).estimates.elpd)
        push!(waics, waic(log_likelihood).estimates.elpd)
        idata = from_mcmcchains(
                    chains[i];
                    log_likelihood=Dict("ll" => log_likelihood),
                    observed_data=(; ind.y),
                    library=Turing,
                    )
        idatas[Symbol((times[i]))] = idata
    end
    idatas = NamedTuple(idatas);
    #comparison = compare(idatas, elpd_method=loo)

    norm_loos = exp.(loos)/sum(exp.(loos));
    plt_loo = bar(times, norm_loos, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="LOO")

    norm_waics = exp.(waics)/sum(exp.(waics));
    plt_waic = bar(times, norm_waics, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="WAIC")

    return norm_loos, norm_waics, plt_loo, plt_waic
end

#####################################################################
#         Option 6: Marginal Likelihood sampled from posterior
# The marginal likelihood is defined as P(x) = ∫p(x|θ)p(θ)dθ
# We will take samples of θ from the posterior (the generated chains)
# and use those θ to approximate the integral by discretizing the sum
#####################################################################

function ml_post(chains, models)
    mls_post = [];
    for i in 1:length(chains)
        samples_posterior = hcat(vcat(chains[i][burnin:end][:D].data...), vcat(chains[i][burnin:end][Symbol("etas[1]")].data...), vcat(chains[i][burnin:end][Symbol("etas[2]")].data...));
        samples_posterior = DataFrame(samples_posterior, ["D", "eta1", "eta2"]);

        n=min(10000, size(samples_posterior,1));
        samples_posterior = samples_posterior[shuffle(1:nrow(samples_posterior))[1:n], :];

        ll = 0;
        for (ix, row) in enumerate(eachrow(samples_posterior))
            #ljoint = logjoint(models[i], (D = row["D"], etas = [row["eta1"], row["eta2"]]))
            #proposal = logpdf(posterior_dose, row["D"]) + logpdf(posterior_eta1, row["eta1"]) + logpdf(posterior_eta2, row["eta2"])
            #ratio += exp(ljoint)
            D = round(row["D"]/250)*250;
            llikelihood = loglikelihood(models[i], (D = D, etas = [row["eta1"], row["eta2"]]))
            lprior = logprior(models[i], (D = D, etas = [row["eta1"], row["eta2"]]))
            ll += exp(llikelihood + lprior)
        end
        #println(ml[end])
        ml = ll/ size(samples_posterior, 1)
        push!(mls_post, ml)
    end
    norm_mls_post = mls_post/sum(mls_post);
    plt = bar(times, norm_mls_post, xticks = times, legend = false, xlabel = "Time of initial dose (h)", title="Sampled from posterior")

    return norm_mls_post, plt
end