struct LayerDef{A, T}
    kernel::A
    pad::A
    stride::T
    pool::A
end

layer1 = LayerDef([9,5], [0,2], 2, [1,2])
layer2 = LayerDef([1,3], [0,1], 2, [1,2])
layer3 = LayerDef([1,3], [0,1], 2, [1,1])

function create_model_arch(layers...)
    names = []
    for i = 1:length(layers)
        push!(names, "layer"*string(i))
    end
    n = tuple(Symbol.(names)...)
    return NamedTuple{n}(layers)
end

function output_dims(indims::Array{N}, kerndims::Array{N}, pad::Array{N}, stride::N, pool::Array{N}) where N <: Int
    Int.(floor.((((indims - kerndims + 2 .* pad) ./ stride) .+ 1) ./ pool))
end

function calculate_output_size(input_dims::Array{N}) where N <: Int
    l1 = output_dims(input_dims, [input_dims[1], 5], [0,2], 2, [1, 2]) 
    @info l1
    l2 = output_dims(l1, [1,3], [0,1], 2, [1, 2]) 
    @info l2
    l3 = output_dims(l2, [1,3], [0,1], 2, [1, 1])
    @info l3
end

function calculate_final_outdims(indims, arch)
    @info "Layer 0 dims: " * string(indims)
    for (i, k) in enumerate(arch)
        indims = output_dims(indims, k.kernel, k.pad, k.stride, k.pool)
        @info "Layer " * string(i) * " dims: " * string(indims)
    end
    return indims
end

function create_model(input_size, arch)
    Conv(tuple(arch.layer1...), input_size[3]=>16, stride=2, pad=(0,2), swish),
    MaxPool((1,2)),
    Conv((1, 3), 16=>32, pad=(0,1), stride=2, swish),
    MaxPool((1,2)),
    Conv((1,3), 32=>64, pad=(0,1), stride=2, swish),
    flatten,
    Dense(latent_size, nclasses))