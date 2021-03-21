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

mutable struct Model
    model1
    model2
    model3
end

function (M::Model)(x)
    return M.model1(x), M.model2(x), M.model3(x)
end

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

function build_Model(input_size::Array{T,1}, archs::Array, nclasses::Int ; silent::Bool = false) where T<:Int  
    out = []
    for arch in archs
        push!(out, build_model(input_size, arch, nclasses; silent))
    end
    Model(out[1], out[2], out[3])
end
