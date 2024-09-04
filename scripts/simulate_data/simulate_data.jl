using DrWatson
@quickactivate "compartment-initialization"

using Plots
using DataFrames
using CSV
using Printf

include(srcdir("bjorkman.jl"));
#include(srcdir("mceneny.jl"));
include(srcdir("create_df_from_I.jl"));

# Individual data
weight = 70
age = 40
ffm = weight * (1-0.3)

# Set dosing callback
D = 1750;
new_dose_time = 24
I = [0 D D*60 1/60; new_dose_time 2000 2000*60 1/60];
#I = [0 D D*60 1/60; 12 D D*60 1/60; new_dose_time 2000 2000*60 1/60; new_dose_time + 12 2000 2000*60 1/60];

cb = generate_dosing_callback(I);
ind = Individual((weight = weight, age = age, ffm=ffm), [], [], cb, id="subject_1");

# Population pk
max_time = 72
saveat_pop = collect(0:0.1:max_time);
y_pop = predict_pk_bjorkman(ind, I, saveat_pop; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, max_time));
#y_pop = predict_pk_mceneny(ind, I, saveat_pop; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, max_time));

# Individual pk: includes random effects and residual error
saveat_ind = collect(new_dose_time+1:4:48);

Ω = build_omega_matrix();
etas = sample_etas(Ω)
#etas = zeros(2)
sigma = 1

y_ind = predict_pk_bjorkman(ind, I, saveat_ind; save_idxs=[1], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, max_time));
#y_ind = predict_pk_mceneny(ind, I, saveat_ind; save_idxs=[1], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, max_time));

# Plot population model and measurements
plt = plot(saveat_pop, y_pop, label="Population model");
scatter!(plt, saveat_ind, y_ind, color="red", label="Observed values")


# The time of the first dose is unknown, so the observed values are shifted so that t=0 is the moment of the second dose
observed_times = saveat_ind .- new_dose_time;
plt2 = scatter(observed_times, y_ind, color="red", label="Observed values")

p = plot(plt, plt2, layout=(2,1))
display(p)

# Number of doses before the recording time
num_doses_rec = 3;

# Reconstruct dosing scheme from the recording time on. The new dose is at t=0 now.
I_ = I[num_doses_rec:end,:];
I_[:,1] = I_[:,1] .- I_[1,1];

df = create_df_from_I(ind, y_ind, observed_times, I_, etas);

boolean_etas = all(etas .== 0) ? "n" : "y";

#CSV.write(datadir("exp_raw", "bjorkman_multi_sigma=$(sigma)_etas=$(boolean_etas).csv"), df);
#savefig(p, plotsdir("bjorkman_multi_sigma=$(sigma)_etas=$(boolean_etas).png"))
