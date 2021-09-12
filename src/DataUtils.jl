module DataUtils

using Flux
using Flux.Data: DataLoader
using Random, DelimitedFiles, Distributed
using ProgressMeter: Progress, next!

export get_train_validation, get_test_set

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
    @info "Getting data..."
    out = Array{Array{Float32}}(undef, length(files))
    progress = Progress(length(files))
    for i = 1:length(files)
        # data = permutedims(readdlm(data_dir * "/Inertial_Signals/" * files[i], Float32))
        data = readdlm(data_dir * "/Inertial_Signals/" * files[i], Float32)
        out[i] = data
        next!(progress)
    end
    return create_tensor(out)
end

function get_train_validation(X, Y, batch_size, train_prop, loc, shuffle=true)
    train_prop < 1 ? nothing : error("Validation set must have entries. Please use a train_prop value less than 1.")
    train_prop > 0 ? nothing : error("Training set must have entries. Please use a train_prop value greater than 0.")
    ind = randperm(length(Y))

    if shuffle
        X = X[:,:,:,ind]
        Y = Y[ind]
    end

    classes = sort!(unique(Y))

    idx = 1:Int(floor(train_prop*length(Y)))
    X_train = X[:,:,:,idx] |> loc
    Y_train = Flux.onehotbatch(Y[idx], classes) |> loc
    X_val = X[:,:,:,last(idx)+1:length(Y)] |> loc
    Y_val = Flux.onehotbatch(Y[last(idx)+1:length(Y)], classes) |> loc

    #X_train = reshape(X_train, size(X_train,1), size(X_train,2), 1, size(X_train,3))
    train_set = DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=shuffle)

    #X_val = reshape(X_val, size(X_val, 1), size(X_val,2), 1, size(X_val,3))
    val_set = DataLoader((X_val, Y_val), batchsize=size(Y_val,2), shuffle=shuffle)

    return train_set, val_set
end

function get_test_set(X, Y, loc, shuffle=true)
    classes = sort!(unique(Y))
    X_test = X |> loc
    Y_test = Flux.onehotbatch(reshape(Y, size(Y, 1)), classes) |> loc
    
    #X_test = reshape(X_test, size(X_test,1), size(X_test,2), 1, size(X_test,3))
    test_set = DataLoader((X_test, Y_test), batchsize=size(Y, 2), shuffle=shuffle)

    return test_set
end

end
