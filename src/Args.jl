mutable struct Args
    η::Float32 # learning rate
    ρ::Float32 # regularization paramater (for data augmentation)
    batch_size::Int64 # batch size
    train_prop::Float32 # % of train data to be used for training (validation set is 1 - train_prop)
    epochs::Int64 # number of epochs
    seed::Int64 # random seed
    cuda::Bool  # attempt to use GPU
    input_dims::Array{Int64}  # input size
    nclasses::Int64  # classes
    lr_patience::Int64  # non-improving iterations before learning rate drop
    γ::Float32  # amount to drop lr (1/γ)
    convergence::Int64 # non-improving iterations to quit
    shuffle::Bool #whether to shuffle data before splitting
    scale::Bool #whether to apply Z score scaling
    tblogging::Bool  # use tensorboard
    save_model::Bool
    save_path::String  # results path
    train_dir::String
    testdir::String
    Args() = new(
        1e-3, 1e-1, 32, 0.8, 10, 0, true, [9, 128, 1], 6, 5, 10.0, 10, true, true, false, true, "output_" * Dates.format(now(), "Y-mm-dd-HMS"), DIR*"/../data/train", DIR*"/../data/test"
    )
end