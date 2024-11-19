using DrWatson
@quickactivate "compartment-initialization"

include(srcdir("dose_time_prediction.jl"));

include(srcdir("bjorkman.jl")); # Model that will be used to make predictions
type_prior = "discrete";
times = [24, 48, 72];
metric = "ml_is"; # joint, ml_post, ml_prior, ml_is, loo, waic

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
#df.ffm = df.weight*(1-0.3);
data = df[df.id .== 5, :]; #5, 19

# Run algorithm
norm_metric, plt_metric, chains = dose_time_prediction(type_prior, times, metric, data);

# Sample time, dose, and etas from resulting model
time, dose, etas = sample_dose_time(times, chains, norm_metric)

#preds = [sample_dose_time(times, chains, norm_metric) for i in 1:100];