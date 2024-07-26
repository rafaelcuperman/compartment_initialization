using DrWatson
@quickactivate "compartment-initialization"

using Plots
using DataFrames
using CSV

# Here you may include files from the source directory
include(srcdir("bjorkman.jl"));

# Individual data
weight = 70
age = 40

# Set dosing callback
D = 1750
new_dose_time = 12
I = [0 D D*60 1/60; new_dose_time 2000 2000*60 1/60];

# Population pk
saveat_pop = collect(0:0.1:72);
y_pop = predict_pk_bjorkman(weight, age, I, saveat_pop; save_idxs=[1], σ=0, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72));

# Individual pk: includes random effects and residual error
saveat_ind = collect(new_dose_time+1:8:48);

Ω = build_omega_matrix();
etas = sample_etas(Ω)
etas = zeros(2)
sigma = 5

y_ind = predict_pk_bjorkman(weight, age, I, saveat_ind; save_idxs=[1], σ=sigma, etas=etas, u0=zeros(2), tspan=(-0.1, 72));

# Plot population model and measurements
plt = plot(saveat_pop, y_pop, label="Population model");
scatter!(plt, saveat_ind, y_ind, color="red", label="Observed values")

# The time of the first dose is unknown, so the observed values are shifted so that t=0 is the moment of the second dose
observed_times = saveat_ind .- new_dose_time;
plt2 = scatter(observed_times, y_ind, color="red", label="Observed values")

p = plot(plt, plt2, layout=(2,1))
display(p)

df = DataFrame(id = "subject_1",
               time = [0; observed_times],
               dv = [missing; vec(y_ind)],
               mdv = [0; fill(1, length(y_sim))],
               amt = [I[2,2]; fill(missing, length(y_sim))],
               rate = [I[2,3]; fill(missing, length(y_sim))],
               duration = [I[2,4]; fill(missing, length(y_sim))],
               age = age,
               weight = weight
               )

boolean_etas = all(etas .== 0) ? "n" : "y";

CSV.write(datadir("exp_raw", "bjorkman_sigma=$(sigma)_etas=$(boolean_etas).csv"), df);
