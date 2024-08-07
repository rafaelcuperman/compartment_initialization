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
df = first(df, 30);

# Random effects
Ω = build_omega_matrix();

# Residual
sigma = 5;

# Points where measurements are taken
new_dose_time = 12;
saveat = [];
append!(saveat, collect(new_dose_time+1:1:48));

# Simulate pk curve for all individuals
df_pk = DataFrame()
for i in eachrow(df)
    # Set dosing callback
    D = round(i.weight*25/250)*250; # Dose is weight*25 rounded to the nearest 250
    I = [0 D D*60 1/60; new_dose_time D D*60 1/60];
    cb = generate_dosing_callback(I);

    ind = Individual((weight = i.weight, age = i.age), [], [], cb, id=i.id)
    etas = sample_etas(Ω);

    y = predict_pk_bjorkman(ind, I, saveat; save_idxs=[1], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, 48));
    
    df_pk = vcat(df_pk, create_df_from_I(ind, y, saveat, I, etas))
end

#CSV.write(datadir("exp_pro", "bjorkman_population_1h.csv"), df_pk);