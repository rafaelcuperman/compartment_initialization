using DrWatson
@quickactivate "compartment-initialization"

using DeepCompartmentModels: BasicIndividual, generate_dosing_callback

""" Hemophilia PK model based on https://pubmed.ncbi.nlm.nih.gov/22042695/ without the covariate effects """
function simple_pk()
    CL = 193 #mL/h
    V1 = 2.22 #L
    Q = 147 #mL/h 
    V2 = 0.73 #L
    return CL, V1, Q, V2
end

"""Standardize units to UI, dL, and h """
function standardize_units_simple(CL, V1, Q, V2)
    CL = CL / 100
    V1 = V1 * 10
    Q = Q / 100
    V2 = V2 * 10

    return CL, V1, Q, V2
end

"""Builds Ω matrix for etas"""
function build_omega_matrix_simple()
    ω₁_sq = log((exp(0.1) - 1) + 1)
    ω₂_sq = log((exp(0.1) - 1) + 1)
    ω = 0 * sqrt(ω₁_sq) * sqrt(ω₂_sq)

    return [ω₁_sq ω; ω  ω₂_sq]
end

"""Sample etas from Ω matrix"""
function sample_etas(Ω)
    return rand(MultivariateNormal(zeros(2), Ω), 1)
end

""""Include etas (random effects)"""
function include_etas_simple(CL, V1, Q, V2; etas=zeros(2))
    CL = CL * exp.(etas[1])
    V1 = V1 * exp.(etas[2])

    CL, V1, Q, V2 = standardize_units_simple(CL, V1, Q, V2)

    return CL, V1, Q, V2
end

""" Run model and get pk values""" 
function predict_pk_simple(I::AbstractMatrix, saveat; save_idxs=[1], σ=5, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72))
    CL, V1, Q, V2 = simple_pk()
    CL, V1, Q, V2 = include_etas_simple(CL, V1, Q, V2, etas=etas)

    cb = generate_dosing_callback(I);
    prob = ODEProblem(two_comp!, u0, tspan);

    #https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    # The ODE can become stiff according to simulations, so we use a method to switch when it becomes stiff (AutoTsit5)
    sol = solve(remake(prob, p = [CL, V1, Q, V2, 0.]), AutoTsit5(Rosenbrock23()), saveat=saveat, save_idxs=save_idxs, tstops=cb.condition.times, callback=cb, maxiters=1e6, force_dtmin=true)
    y = hcat(sol.u...)'

    # Add residual error
    y = y .+ rand(Normal(0., σ), size(y))

    # Truncate negative values to 0
    y = max.(y, 0.)

    return y
end;

""" Takes an individual as input"""
function predict_pk_simple(i::BasicIndividual, I::AbstractMatrix, args...; kwargs...)
    predict_pk_simple(I, args...; kwargs...)
end

""" Creates an individual from a df"""
function individual_from_df(df)
    df_ = df[df.mdv .== 0, :]; # Remove dosing rows

    data = Float64.(df_[!, :dv]);
    times = Float64.(df_[!, :time]);
    
    # Reconstruct dosing matrix
    I = Float64.(Matrix(df[df.mdv .== 1, [:time, :amt, :rate, :duration]]));
    cb = generate_dosing_callback(I);
    
    ind = Individual((), times, data, cb);
    return ind, I
end
