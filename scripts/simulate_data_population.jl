using DrWatson
@quickactivate "compartment-initialization"

using Plots
using DataFrames
using CSV
using Printf
using DeepCompartmentModels

include(srcdir("bjorkman.jl"));
include(srcdir("create_df_from_I.jl"));

# Read population data
df = CSV.read(datadir("exp_raw", "dcm/population_data.csv"), DataFrame);

# Set dosing callback
D = 1750;
I = [0 D D*60 1/60];
cb = generate_dosing_callback(I);

# Random effects
Ω = build_omega_matrix();

# Residual
sigma = 5;

# Points where measurements are taken
saveat = [0.5];
append!(saveat, collect(6:6:48));

# Simulate pk curve for all individuals
df_pk = DataFrame()
for i in eachrow(df)
    ind = Individual((weight = i.weight, age = i.age), [], [], cb, id=i.id)
    etas = sample_etas(Ω);
    y = predict_pk_bjorkman(ind, I, saveat; save_idxs=[1], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, 48));
    
    df_pk = vcat(df_pk, create_df_from_I(ind, y, saveat, I, etas))
end

CSV.write(datadir("exp_pro", "bjorkman_population_sigma=$(sigma).csv"), df_pk);