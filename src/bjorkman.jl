using DrWatson
@quickactivate "compartment-initialization"

using DeepCompartmentModels: BasicIndividual, generate_dosing_callback

sigma_additive = 5;
sigma_proportional = 0;

""" Hemophilia PK model based on https://pubmed.ncbi.nlm.nih.gov/22042695/"""
function bjorkman(weight::Real, age::Real)
    CL = @. 193 * (weight/56)^0.8 * (1 - 0.0045*(age - 22)) #mL/h
    V1 = @. 2.22 * (weight/56)^0.95 #L
    Q = 147 #mL/h 
    V2 = @. 0.73 * (weight/56)^0.76 #L
    return CL, V1, Q, V2
end
""" Takes an individual as input"""
function bjorkman(i::BasicIndividual)
    weight = i.x.weight
    age = i.x.age

    return bjorkman(weight, age)
end

"""Standardize units to UI, dL, and h from Bjorkman, et al. model"""
function standardize_units_bjorkman(CL, V1, Q, V2)
    CL = CL / 100
    V1 = V1 * 10
    Q = Q / 100
    V2 = V2 * 10

    return CL, V1, Q, V2
end

"""Builds Ω matrix for etas"""
function build_omega_matrix()
    ω₁_sq = log(0.3^2 + 1)
    ω₂_sq = log(0.21^2 + 1)
    ω = 0.45 * sqrt(ω₁_sq) * sqrt(ω₂_sq)

    return [ω₁_sq ω; ω  ω₂_sq]
end

"""Sample etas from Ω matrix"""
function sample_etas(Ω)
    return rand(MultivariateNormal(zeros(2), Ω), 1)
end

""""Include etas (random effects) according to Bjorkman, et al. model"""
function include_etas_bjorkman(CL, V1, Q, V2; etas=zeros(2))
    CL = CL * exp.(etas[1])
    V1 = V1 * exp.(etas[2])

    CL, V1, Q, V2 = standardize_units_bjorkman(CL, V1, Q, V2)

    return CL, V1, Q, V2
end

""" Run model and get pk values""" 
function predict_pk_bjorkman(weight::Real, age::Real, I::AbstractMatrix, saveat; save_idxs=[1], σ=8.9, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72))
    CL, V1, Q, V2 = bjorkman(weight, age);
    CL, V1, Q, V2 = include_etas_bjorkman(CL, V1, Q, V2, etas=etas)

    cb = generate_dosing_callback(I);
    prob = ODEProblem(two_comp!, u0, tspan);

    #https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    # The ODE can become stiff according to simulations, so we use a method to switch when it becomes stiff (AutoTsit5)
    sol = solve(remake(prob, p = [CL, V1, Q, V2, 0.]), AutoTsit5(Rosenbrock23()), saveat=saveat, save_idxs=save_idxs, tstops=cb.condition.times, callback=cb, maxiters=1e6, force_dtmin=true)
    y = hcat(sol.u...)'

    if σ != 0
        # Add residual error
        y = y + rand(Normal(0., σ), size(y))
    end

    # Truncate negative values to 0
    y = max.(y, 0.)

    return y
end;

""" Takes an individual as input"""
function predict_pk_bjorkman(i::BasicIndividual, I::AbstractMatrix, args...; kwargs...)
    weight = i.x.weight
    age = i.x.age

    predict_pk_bjorkman(weight, age, I, args...; kwargs...)
end

predict_pk(args...; kwargs...) = predict_pk_bjorkman(args...; kwargs...);

""" Creates an individual from a df"""
function individual_from_df(df)
    df_ = df[df.mdv .== 0, :]; # Remove dosing rows

    data = Float64.(df_[!, :dv]);
    times = Float64.(df_[!, :time]);
    
    age = df[1, :age];
    weight = df[1, :weight];
    
    # Reconstruct dosing matrix
    I = Float64.(Matrix(df[df.mdv .== 1, [:time, :amt, :rate, :duration]]));
    cb = generate_dosing_callback(I);
    
    ind = Individual((weight = weight, age = age), times, data, cb);
    return ind, I
end