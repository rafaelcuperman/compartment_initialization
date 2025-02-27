using DrWatson
@quickactivate "compartment-initialization"

include(srcdir("dose_time_prediction.jl"));

include(srcdir("bjorkman.jl")); # Model that will be used to make predictions
use_etas = true;
type_prior = "continuous";
times = [24, 48, 72];
metric = "ml_is"; # joint, ml_post, ml_prior, ml_is, loo, waic

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "mceneny_population_1h.csv"), DataFrame);
#df.ffm = df.weight*(1-0.3);
data = df[df.id .== 5, :]; #5, 19

# Run algorithm
norm_metric, plt_metric, chains = dose_time_prediction(type_prior, times, metric, data; use_etas=use_etas);

### Sample time, dose, and etas from resulting model
time, dose, etas = sample_dose_time(times, chains, norm_metric; use_etas=use_etas)



### Sample n solutions
n=100;
preds = [sample_dose_time(times, chains, norm_metric) for i in 1:n];
plt = plot(title="n = $n");
for i in preds
    ind, I = individual_from_df_general(data);
    I0 = [0 i[2] i[2]*60 1/60];
    I[1] = i[1];
    I = vcat(I0, I);

    etas = i[3];

    max_t = maximum(ind.t) + i[1];
    saveat = 0:1:max_t+10;
    y = predict_pk(ind, I, saveat; save_idxs=[1], σ=0, etas=etas, u0=zeros(2), tspan=(-0.1, max_t+10));

    color = :black
    #if i[1] == 72
    #    color = :black
    #elseif i[1] == 48
    #    color = :black
    #elseif i[1] == 24
    #    color = :black
    #else
    #    color = :gray
    #end
    plot!(plt, collect(saveat) .- i[1], y, color=color, alpha=0.1, label=nothing)
end
display(plt)
ind, I = individual_from_df_general(data);
scatter!(plt, ind.t, ind.y, color=:red, markersize=2, label=nothing)