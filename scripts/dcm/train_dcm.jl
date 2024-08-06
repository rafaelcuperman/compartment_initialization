using DrWatson
@quickactivate "compartment-initialization"

import Optimisers
import CSV
using DataFrames
using DeepCompartmentModels
using JLD2
using Plots

# Read population data
df = CSV.read(datadir("exp_pro", "bjorkman_population_sigma=5.csv"), DataFrame);
select!(df, Not(:etas));

# Capitalize names of columns
new_names = [uppercase(string(col)) for col in names(df)]
rename!(df, Symbol.(new_names))

population = DeepCompartmentModels.load(df, [:WEIGHT, :AGE]);

ann = Chain(
    # Our data set contains two covariates, which we feed into a hidden layer with 16 neurons
    Dense(2, 16, relu), 
    Dense(16, 4, softplus), # Our differential equation has four parameters
)

model = DCM(two_comp!, 2, ann);

function my_callback(epoch, loss)
    if epoch % 10 == 0
        println("Epoch $epoch, loss $loss")
    end
end

fit!(model, population, Optimisers.Adam(0.01), 500, callback=my_callback)

# Save model
#jldsave(datadir("sims", "dcm/mymodel.jld2"); model)

# Make prediction
#y, _ = predict(model, population[1], saveat=0:6:48);
