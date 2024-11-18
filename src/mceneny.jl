using DrWatson
@quickactivate "compartment-initialization"

using DeepCompartmentModels: BasicIndividual, generate_dosing_callback

sigma_additive = 0;
sigma_proportional = 0.17;

# Fat free mass = weight * (1-bodyfat)
# Assume 30% bodyfat

""" Hemophilia PK model based on https://ascpt.onlinelibrary.wiley.com/doi/epdf/10.1002/cpt.3203"""
function mceneny(ffm::Real, age::Real)
    CL = @. 0.238 * (ffm/53)^0.794 * (1-0.205*max(0, (age-21)/21))
    V1 = @. 3.01 * (ffm/53)^1.02
    Q = 0.142 
    V2 = 0.525 * (ffm/53)^0.787
    return CL, V1, Q, V2
end

""" Takes an individual as input"""
function mceneny(i::BasicIndividual)
    ffm = i.x.ffm
    age = i.x.age

    return mceneny(ffm, age)
end

"""Builds Ω matrix for etas"""
function build_omega_matrix()
    ω₁_sq = log(0.411^2 + 1)
    ω₂_sq = log(0.324^2 + 1)
    ω = 0.703 * sqrt(ω₁_sq) * sqrt(ω₂_sq)

    return [ω₁_sq ω; ω  ω₂_sq]
end

"""Sample etas from Ω matrix"""
function sample_etas(Ω)
    return rand(MultivariateNormal(zeros(2), Ω), 1)
end

""""Include etas (random effects) according to Bjorkman, et al. model"""
function include_etas_mceneny(CL, V1, Q, V2; etas=zeros(2))
    CL = CL * exp.(etas[1])
    V1 = V1 * exp.(etas[2])

    return CL, V1, Q, V2
end


""" Run model and get pk values""" 
function predict_pk_mceneny(ffm::Real, age::Real, I::AbstractMatrix, saveat; save_idxs=[1], σ=0.17, etas=zeros(2), u0=zeros(2), tspan=(-0.1, 72))
    CL, V1, Q, V2 = mceneny(ffm, age);
    CL, V1, Q, V2 = include_etas_mceneny(CL, V1, Q, V2, etas=etas)

    cb = generate_dosing_callback(I);
    prob = ODEProblem(two_comp!, u0, tspan);

    #https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    # The ODE can become stiff according to simulations, so we use a method to switch when it becomes stiff (AutoTsit5)
    sol = solve(remake(prob, p = [CL, V1, Q, V2, 0.]), AutoTsit5(Rosenbrock23()), saveat=saveat, save_idxs=save_idxs, tstops=cb.condition.times, callback=cb, maxiters=1e7, force_dtmin=true)
    y = hcat(sol.u...)'

    if σ != 0
        # Add residual error
        ϵ = rand(Normal(0, σ), size(y))
        y = y .* (1 .+ ϵ)
    end

    # Truncate negative values to 0
    y = max.(y, 0.)

    # Convert to UI/dL
    y = y./10

    return y
end;

""" Takes an individual as input"""
function predict_pk_mceneny(i::BasicIndividual, I::AbstractMatrix, args...; kwargs...)
    ffm = i.x.ffm
    age = i.x.age

    predict_pk_mceneny(ffm, age, I, args...; kwargs...)
end

predict_pk(args...; kwargs...) = predict_pk_mceneny(args...; kwargs...);


""" Creates an individual from a df"""
function individual_from_df(df)
    df_ = df[df.mdv .== 0, :]; # Remove dosing rows

    data = Float64.(df_[!, :dv]);
    times = Float64.(df_[!, :time]);
    
    age = df[1, :age];
    ffm = df[1, :ffm];
    
    # Reconstruct dosing matrix
    I = Float64.(Matrix(df[df.mdv .== 1, [:time, :amt, :rate, :duration]]));
    cb = generate_dosing_callback(I);
    
    ind = Individual((ffm = ffm, age = age), times, data, cb);
    return ind, I
end