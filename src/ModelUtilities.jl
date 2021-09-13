using Flux
export LayerDef, create_model_arch, output_dims, calculate_final_outdims, create_model

mutable struct LayerDef{T, P, A}
    kernel::T
    channels::P
    pad::T
    stride::A
    pool::T
    LayerDef(kernel::Tuple, in_channels::Int, out_channels::Int, pad::Tuple, stride::Int, pool::Tuple) = new{Tuple, Pair, Int}(
        kernel,
        Pair(in_channels, out_channels),
        pad,
        stride,
        pool
    )
end

mutable struct Model{T}
    paths::T
end

Model(paths...) = Model(paths)

Flux.@functor Model

(m::Model)(x::AbstractArray) = map(f -> f(x), m.paths)

function create_model_arch(layers::LayerDef...)::NamedTuple
    names = []
    for i = 1:length(layers)
        push!(names, "layer"*string(i))
    end
    n = tuple(Symbol.(names)...)
    return NamedTuple{n}(layers)
end

function output_dims(indims::Array{T,1}, kerndims::Tuple{T,T}, pad::Tuple{T,T}, stride::T, pool::Tuple{T,T}) where T<:Int
    Int.(floor.((((indims .- kerndims .+ 2 .* pad) ./ stride) .+ 1) ./ pool))
end

function calculate_final_outdims(indims::Array{T,1}, arch::NamedTuple, silent::Bool) where T<:Int
    silent || @info "Input dims: " * string(indims)
    for (i, k) in enumerate(arch)
        indims = output_dims(indims[1:2], k.kernel, k.pad, k.stride, k.pool)
        push!(indims, k.channels[2])
        silent || @info "Layer " * string(i) * " output dims: " * string(indims)
    end
    return prod(indims)
end

function make_layer(layers::Array, layer::LayerDef)
    push!(layers, Conv(tuple(layer.kernel...), layer.channels, stride=layer.stride, pad=layer.pad, swish))
    sum(layer.pool .> 1) > 0 && push!(layers, MaxPool(layer.pool))
end

function build_model(input_size::Array{T,1}, arch::NamedTuple, nclasses::Int ; silent::Bool = false) where T<:Int
    latent_size = calculate_final_outdims(input_size, arch, silent)
    l = []
    for i = 1:length(arch)
        make_layer(l, arch[i])
    end
    push!(l, flatten, Dense(latent_size, nclasses))
    return Chain(l...)
end

function build_Model(input_size::Array{T,1}, archs::Array, nclasses::Int, loc ; silent::Bool = false) where T<:Int  
    out = []
    for arch in archs
        push!(out, build_model(input_size, arch, nclasses; silent))
    end
    Model(tuple(out...) |> loc)
end