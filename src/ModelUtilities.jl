using Flux
export LayerDef, create_model_arch, output_dims, calculate_final_outdims, create_model

mutable struct Model{T}
    paths::T
end

Model(paths...) = Model(paths)

Flux.@functor Model

(m::Model)(x::AbstractArray) = map(f -> f(x), m.paths)

function build_Model(path, input_dims, nclasses)
    model_defs = JSON.parse(String(read(path)))
    out = []
    for key in keys(model_defs)
        m = eval(Meta.parse(model_defs[key][1]))
        latent_dims = Flux.outputsize(m, tuple(vcat(input_dims, 1)...))
        push!(out, Chain(m, flatten, Dense(prod(latent_dims), nclasses)))
    end
    return Model(tuple(out...))
end