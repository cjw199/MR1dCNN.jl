module ModelUtilities

using Flux
export LayerDef, create_model_arch, output_dims, calculate_final_outdims, create_model

struct LayerDef
    kernel
    channels
    pad
    stride
    pool
    LayerDef(kernel, in_channels, out_channels, pad, stride, pool) = new(
        kernel,
        Pair(in_channels, out_channels),
        pad,
        stride,
        pool
    )
end

function create_model_arch(layers...)
    names = []
    for i = 1:length(layers)
        push!(names, "layer"*string(i))
    end
    n = tuple(Symbol.(names)...)
    return NamedTuple{n}(layers)
end

function output_dims(indims, kerndims, pad, stride, pool)
    Int.(floor.((((indims .- kerndims .+ 2 .* pad) ./ stride) .+ 1) ./ pool))
end

function calculate_final_outdims(indims, arch, silent)
    silent || @info "Input dims: " * string(indims)
    for (i, k) in enumerate(arch)
        indims = output_dims(indims[1:2], k.kernel, k.pad, k.stride, k.pool)
        push!(indims, k.channels[2])
        silent || @info "Layer " * string(i) * " output dims: " * string(indims)
    end
    return prod(indims)
end

function make_layer(layers, layer)
    push!(layers, Conv(tuple(layer.kernel...), layer.channels, stride=layer.stride, pad=layer.pad, swish))
    sum(layer.pool .> 1) > 0 && push!(layers, MaxPool(layer.pool))
end

function build_model(input_size, arch, nclasses ; silent = false)
    latent_size = calculate_final_outdims(input_size, arch, silent)
    l = []
    for i = 1:length(arch)
        make_layer(l, arch[i])
    end
    push!(l, flatten, Dense(latent_size, nclasses))
    return Chain(l...)
end

end #module