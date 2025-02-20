using DrWatson
@quickactivate "compartment-initialization"

include(srcdir("dose_time_prediction.jl"));

include(srcdir("bjorkman.jl")); # Model that will be used to make predictions

between_dose = 1; #Time between dose for measurments used for MCMC
use_etas = true;
type_prior = "continuous";
times = [24, 48, 72];
metric = "ml_is"; # joint, ml_post, ml_prior, ml_is, loo, waic

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "mceneny_population_1h.csv"), DataFrame);
ids = unique(df.id);

times_real = [];
probs = [];

loop_ids = ids[1:50];
for id in loop_ids
    println("----------------Patient $id/$(length(loop_ids))----------------")
    data = df[df.id .== id, :];
    data = filter(row -> (row.time % between_dose == 0) .| (row.time == 1), data);

    metadata = eval(Meta.parse(data[1,:metadata]));
    time_real = metadata["time"];

    push!(times_real, time_real);

    # Run algorithm
    norm_metric, _, _ = dose_time_prediction(type_prior, times, metric, data; use_etas=use_etas, num_chains=1, plot_metric=false);

    push!(probs, norm_metric)

end
println("---------------------------DONE-------------------------------")

# Get indices of correct times
times_indices = indexin(times_real, times);

# Get predicted probability of correct times
prob_correct = [probs[i][times_indices[i]] for i in eachindex(times_indices)];

# Remove predicted probability of correct time
prob_incorrect = [probs[i][setdiff(1:length(probs[i]), times_indices[i])] for i in eachindex(times_indices)];

# Get the probabilities of the time with largest probability among the incorrect times
prob_first = [maximum(v) for v in prob_incorrect];

# Get the probabilities of the time with smallest probability among the incorrect times
prob_second = [minimum(v) for v in prob_incorrect];

# Join the three probabilities in a matrix. First column is the probability of the correct time, second column is the probability of the time with largest probability among the incorrect times, and third column is the probability of the time with smallest probability among the incorrect times
joined_probs = hcat(prob_correct, prob_first, prob_second);

# Mean of each column
means = mean(joined_probs, dims=1);
#stds = std(joined_probs, dims=1);

# Plot means
plt = bar(means',
        #yerr=stds', 
        xticks=(1:length(means), ["Correct time", "Wrong time 1", "Wrong time 2"]), 
        legend=nothing
        );
display(plt);


#CSV.write("population_continuous.csv", DataFrame(hcat(hcat(probs...)', times_real, times_indices), ["24", "48", "72", "real_time", "real_index"]))