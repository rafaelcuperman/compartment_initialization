using DataFrames

function create_df_from_I(ind, ys, ts, I, metadata)
    df_ = DataFrame(id = ind.id,
                time = vcat([ts; I[:,1]]...),
                dv = vcat([vec(ys); fill(missing, size(I,1))]...),
                mdv = vcat([fill(0, length(ys)), fill(1, size(I,1))]...),
                amt = vcat([fill(missing, length(ys)), I[:,2]]...),
                rate = vcat([fill(missing, length(ys)), I[:,3]]...),
                duration = vcat([fill(missing, length(ys)), I[:,4]]...),
                age = ind.x.age,
                weight = ind.x.weight,
                metadata = metadata,
                )

    sort!(df_, [:id, :time])
    return df_
end;