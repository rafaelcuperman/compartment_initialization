using DrWatson
@quickactivate "compartment-initialization"

using Plots
using DataFrames
using CSV
using Printf
using DeepCompartmentModels

#include(srcdir("bjorkman.jl"));
include(srcdir("mceneny.jl"));
include(srcdir("create_df_from_I.jl"));

# Read population data
df = CSV.read(datadir("exp_raw", "dcm/population_data.csv"), DataFrame);
df = first(df, 50); # First n patients

# Build FFM from weight
df.ffm = df.weight .* (1-0.3);

# Random effects
Ω = build_omega_matrix();

# Residual
sigma = 0.17;

# Distribution for time of initial dose
dist_times = Categorical([0.2, 0.5, 0.3]); #[24h, 48h, 72h] 

# Simulate pk curve for all individuals
df_pk = DataFrame();
for i in eachrow(df)
    # Time of initial dose
    new_dose_time = rand(dist_times, 1)[1] * 24;

    # Points where measurements are taken
    saveat = [];
    append!(saveat, collect(new_dose_time+1:1:120));
    observed_times = saveat .- new_dose_time;

    # Set dosing callback
    D = round(i.weight*25/250)*250; # Dose is weight*25 rounded to the nearest 250
    I = [0 D D*60 1/60; new_dose_time D D*60 1/60];
    cb = generate_dosing_callback(I);
    num_doses_rec = 2; # Index from where the doses are recorded

    # Build individual
    ind = Individual((weight = i.weight, age = i.age,  ffm = ffm), [], [], cb, id=i.id)
    etas = sample_etas(Ω);

    # Run PK model
    #y = predict_pk_bjorkman(ind, I, vcat(new_dose_time, saveat); save_idxs=[1, 2], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, 120));

    y = predict_pk_mceneny(ind, I, vcat(new_dose_time, saveat); save_idxs=[1, 2], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, 120));

    # Save u0s before new_dose
    u0s = y[1,:];

    # Save PK measurements
    y = y[2:end,1];

    # Reconstruct dosing scheme from the recording time on. The new dose is at t=0 now.
    I_ = I[num_doses_rec:end,:];
    I_[:,1] = I_[:,1] .- I_[1,1];
    
    # Metadata
    metadata = Dict("etas" => etas, 
                    "u0s" => u0s,
                    "time" => new_dose_time,
                    "dose" => D,
                    "sigma" => sigma 
                    )

    df_pk = vcat(df_pk, create_df_from_I(ind, y, observed_times, I_, metadata))
end

CSV.write(datadir("exp_pro", "variable_times", "mceneny_population_1h.csv"), df_pk);