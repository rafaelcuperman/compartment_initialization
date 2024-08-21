using DrWatson
@quickactivate "compartment-initialization"

using DataFrames
using CSV
using StatsPlots
using Turing

include(srcdir("bjorkman.jl"));

# Read data
df = CSV.read(datadir("exp_pro", "variable_times", "bjorkman_population_1h.csv"), DataFrame);
df = df[df.id .== 2, :];

ind, I = individual_from_df(df);

# Choose pk model
pkmodel(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

Ω = build_omega_matrix();

@model function model_dirichlet(pkmodel, ind, I, dose, args...; kwargs...)
    w1 ~ LogNormal(1,1)
    w2 ~ LogNormal(1,1)
    w3 ~ LogNormal(1,1)

    weights = rand(Dirichlet([w1, w2, w3]), 1);
    #weights ~ Dirichlet([w1, w2, w3])
    #weights ~ Dirichlet([1, 1, 1])

    t ~ Truncated(MixtureModel(map(u -> Normal(u, 5), 24:24:72), [weights[1], weights[2], weights[3]]), 0, 96);

    #D ~ Truncated(Normal(I[2], 250), 1000, 5000);
    D = I[2]

    etas ~ MultivariateNormal(zeros(2), Ω);

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
model = model_dirichlet(pkmodel, ind, I, Int(I[2]));

# Sample from model
#initial_params = FillArrays.fill([(weights = [0.2, 0.5, 0.3], t = 72, etas = [0,0])], 3)
chain = sample(model, NUTS(0.65), MCMCSerial(), 1000, 3; progress=true);
plot(chain)

metadata = eval(Meta.parse(df[1,:metadata]))



round_w = 0.01;
[mode(round.((chain[:w1].data) ./round_w).* round_w), mode(round.((chain[:w2].data) ./round_w).* round_w), mode(round.((chain[:w3].data) ./round_w).* round_w)]

[mode(round.((chain[Symbol("weights[1]")].data) ./round_w).* round_w), mode(round.((chain[Symbol("weights[2]")].data) ./round_w).* round_w), mode(round.((chain[Symbol("weights[3]")].data) ./round_w).* round_w)]
