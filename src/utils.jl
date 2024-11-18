using DeepCompartmentModels
function individual_from_df_general(df; covariates=["age", "weight"])
    df_ = df[df.mdv .== 0, :]; # Remove dosing rows

    data = Float64.(df_[!, :dv]);
    times = Float64.(df_[!, :time]);
    
    covs = tuple(df_[1,covariates]...);
    x = NamedTuple((Symbol(covariates[i]) => covs[i] for i in 1:length(covariates)))
    
    # Reconstruct dosing matrix
    I = Float64.(Matrix(df[df.mdv .== 1, [:time, :amt, :rate, :duration]]));
    cb = generate_dosing_callback(I);
    
    ind = Individual(x, times, data, cb);
    return ind, I
end