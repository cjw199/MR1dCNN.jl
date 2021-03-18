module DataUtils

using Flux
using Flux.Data: DataLoader
using Random, DelimitedFiles, Distributed
using ProgressMeter: Progress, next!

export get_train_validation, get_test_set

function data_prep(data_dir)
    files = readdir(data_dir * "/Inertial_Signals")
    @info "Getting data..."
    out = Array{Array{Float32}}(undef, length(files))
    progress = Progress(length(files))
    for i = 1:length(files)
        data = readdlm(data_dir * "/Inertial_Signals/" * files[i], Float32)
        #push!(out, data)
        out[i] = data
        next!(progress)
    end
    return Flux.unsqueeze(permutedims(cat(out..., dims=3), [3,2,1]), 3)
end

function get_train_validation(X, Y, batch_size, train_prop, device, shuffle=true)
    train_prop < 1 ? nothing : error("Validation set must have entries. Please use a train_prop value less than 1.")
    train_prop > 0 ? nothing : error("Training set must have entries. Please use a train_prop value greater than 0.")
    ind = randperm(length(Y))

    if shuffle
        X = X[:,:,:,ind]
        Y = Y[ind]
    end

    classes = sort!(unique(Y))

    idx = 1:Int(floor(train_prop*length(Y)))
    X_train = X[:,:,:,idx] |> device
    Y_train = Flux.onehotbatch(Y[idx], classes) |> device
    X_val = X[:,:,:,last(idx)+1:length(Y)] |> device
    Y_val = Flux.onehotbatch(Y[last(idx)+1:length(Y)], classes) |> device

    #X_train = reshape(X_train, size(X_train,1), size(X_train,2), 1, size(X_train,3))
    train_set = DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=shuffle)

    #X_val = reshape(X_val, size(X_val, 1), size(X_val,2), 1, size(X_val,3))
    val_set = DataLoader((X_val, Y_val), batchsize=size(Y_val,2), shuffle=shuffle)

    return train_set, val_set
end

function get_test_set(X, Y, device, shuffle=true)
    classes = sort!(unique(Y))
    X_test = X |> device
    Y_test = Flux.onehotbatch(reshape(Y, size(Y, 1)), classes) |> device
    
    X_test = reshape(X_test, size(X_test,1), size(X_test,2), 1, size(X_test,3))
    test_set = DataLoader((X_test, Y_test), batchsize=size(Y, 2), shuffle=shuffle)

    return test_set
end

end
