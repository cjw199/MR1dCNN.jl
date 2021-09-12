module DataUtils

using Base: StridedFastContiguousSubArray
using Flux
using Flux.Data: DataLoader
using Random, DelimitedFiles, Distributed
using ProgressMeter: Progress, next!
using StatsBase, Statistics

export get_train_validation, get_test_set

function scale_tensor(X)
    l, m, _, n = size(X)
    StatsBase.fit(ZScoreTransform, reshape(X, l, m*n))
end

function transform_tensor!(Transform, X)
    l, m, _, n = size(X)
    StatsBase.transform!(Transform, reshape(X, l, m*n))
end

function create_tensor(X)
    l = length(X)
    m, n = size(X[1])
    out = Array{eltype(X[1])}(undef, l, n, 1, m)
    for i = 1:l
        for j = 1:m
            out[i, :, :, j] = X[i][j,:]
        end
    end
    return out
end

function data_prep(data_dir)
    files = readdir(data_dir * "/Inertial_Signals")
    @info "Processing data..."
    out = Array{Array{Float32}}(undef, length(files))
    progress = Progress(length(files))
    for i = 1:length(files)
        out[i] = readdlm(data_dir * "/Inertial_Signals/" * files[i], Float32)
        next!(progress)
    end
    return create_tensor(out)
end

function get_train_validation(X, Y, batch_size, train_prop, loc, scale=true, shuffle=true)
    train_prop < 1 ? nothing : error("Validation set must have entries. Please use a train_prop value less than 1.")
    train_prop > 0 ? nothing : error("Training set must have entries. Please use a train_prop value greater than 0.")

    if shuffle
        ind = randperm(length(Y))
    else
        ind = 1:length(Y)
    end

    T = scale_tensor(X)
    if scale
        transform_tensor!(T, X)
    end

    classes = sort!(unique(Y))

    idx = 1:Int(floor(train_prop*length(Y)))
    X_train = X[:,:,:,ind][:,:,:,idx] |> loc
    Y_train = Flux.onehotbatch(Y[ind][idx], classes) |> loc
    X_val = X[:,:,:,ind][:,:,:,last(idx)+1:length(Y)] |> loc
    Y_val = Flux.onehotbatch(Y[ind][last(idx)+1:length(Y)], classes) |> loc

    train_set = DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=shuffle)

    val_set = DataLoader((X_val, Y_val), batchsize=size(Y_val,2), shuffle=shuffle)

    return train_set, val_set, T
end

function get_test_set(X, Y, T, loc, scale, shuffle)
    classes = sort!(unique(Y))
    if scale
        transform_tensor!(T, X)
    end
    X_test = X |> loc
    Y_test = Flux.onehotbatch(reshape(Y, size(Y, 1)), classes) |> loc
    
    #X_test = reshape(X_test, size(X_test,1), size(X_test,2), 1, size(X_test,3))
    test_set = DataLoader((X_test, Y_test), batchsize=size(Y, 2), shuffle=shuffle)

    return test_set
end

end
