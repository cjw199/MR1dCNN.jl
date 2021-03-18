module OneDCNN

const DIR = @__DIR__
using Pkg
Pkg.activate(DIR*"/..")
Pkg.status()

@info "Loading modules..."
using BSON
using CUDA
using Flux
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random, Dates, DelimitedFiles

#load data utlities for wrangling datasets
include("DataUtils.jl")

export Args, train, args

mutable struct Args
    η::Float32 # learning rate
    ρ::Float32 # regularization paramater (for data augmentation)
    batch_size::Int64 # batch size
    train_prop::Float32 # % of train data to be used for training (validation set is 1 - train_prop)
    epochs::Int64 # number of epochs
    seed::Int64 # random seed
    cuda::Bool  # attempt to use GPU
    input_dims::Array{Int64}  # input size
    output_dim::Tuple{Int64, Int64, Int64}  # output size (final dims of conv layers)
    nclasses::Int64  # classes
    lr_patience::Int64  # non-improving iterations before learning rate drop
    γ::Float32  # amount to drop lr (1/γ)
    convergence::Int64 # non-improving iterations to quit
    tblogging::Bool  # use tensorboard
    save_model::Bool
    save_path::String  # results path
    train_dir::String
    testdir::String
    Args() = new(
        1e-3, 0.1, 32, 0.8, 10, 0, true, [9, 128, 1], (1,1,1), 6, 5, 10.0, 10, true, true, "output_" * Dates.format(now(), "Y-mm-dd-HMS"), DIR*"/../data/train", DIR*"/../data/test"
    )
end

args = Args()

function output_dims(indims::Array{N}, kerndims::Array{N}, pad::Array{N}, stride::N, pool::Array{N}) where N <: Int
    Int.(floor.((((indims - kerndims + 2 .* pad) ./ stride) .+ 1) ./ pool))
end

function calculate_output_size(input_dims::Array{N}) where N <: Int
    l1 = output_dims(input_dims, [input_dims[1], 5], [0,2], 2, [1, 2]) 
    l2 = output_dims(l1, [1,3], [0,1], 2, [1, 2]) 
    l3 = output_dims(l2, [1,3], [0,1], 2, [1, 1])
end

function build_model(input_size, nclasses)
    latent_size = prod(calculate_output_size(input_size[1:2])) * 64
    return Chain(
    Conv((input_size[1], 5), input_size[3]=>16, stride=2, pad=(0,2), swish),
    MaxPool((1,2)),
    Conv((1, 3), 16=>32, pad=(0,1), stride=2, swish),
    MaxPool((1,2)),
    Conv((1,3), 32=>64, pad=(0,1), stride=2, swish),
    flatten,
    Dense(latent_size, nclasses))
end

function cast_param(x::A, param::T) where A <:AbstractArray{T} where T <: Real
    convert(eltype(x), param)
end

rng = MersenneTwister(args.seed)

function augment(x::A, ρ::T, device) where A<:AbstractArray{T} where T<:Real
    if device == gpu
        x .+ ρ * CUDA.randn(eltype(x), size(x))
    else
        x .+ ρ * randn(eltype(x), size(x))
    end
end

# training loss 
function model_loss(x::A, y::B, model::C, ρ::T, device) where {C<:Chain, A<:AbstractArray, B<:AbstractArray, T<:Real}
    l = logitcrossentropy(model(augment(x, ρ, device)), y)
end

# loss over data
function total_loss(data::DataLoader, model::Chain)
    l = 0f0
    for (x, y) in data
        l += logitcrossentropy(model(x), y)
    end
    l = l/length(data)
    return l
end

#accuracy over data
function accuracy(data::DataLoader, model::Chain)
    acc = 0
    for (x, y) in data
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) * 1 / size(x,4)
    end
    acc/length(data)
end

# convert model parameters to a suitable data structure for TensorBoard logging
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

# TensorBoard callback function to log model data and loss every epoch
function TBCallback(logger, train_data, val_data, model)
  param_dict = Dict{String, Any}()
  fill_param_dict!(param_dict, model, "")
  with_logger(logger) do
    @info "model" params=param_dict log_step_increment=0
    @info "train" loss=total_loss(train_data, model) acc=accuracy(train_data, model) log_step_increment=0
    @info "validation" loss=total_loss(val_data, model) acc=accuracy(val_data, model)
  end
end

function training_function(model::Flux.Chain, Xs::Flux.Data.DataLoader, params::Flux.Zygote.Params, opt::ADAM, ρ::N, device, progress) where {N<:Real}
    for (xs, ys) in Xs
        train_loss, back = Flux.pullback(() -> model_loss(xs, ys, model, ρ, device), params)
        grad = back(one(train_loss))
        Flux.Optimise.update!(opt, params, grad)
        next!(progress; showvalues=[(:Loss, train_loss)])
    end
end

## model burn-in
@info "Compiling model"
tm = build_model([9,128,1], 3)
d = DataLoader((rand(Float32, 9, 128, 1, 1), onehotbatch([2], [1,2,3]))) 
training_function(tm, d, params(tm), ADAM(1e-3), Float32(.1), cpu, Progress(1))

@info "Ready. Use fields in 'args' atruct to change parameter settings."

# training function
function train(args)
    #args = Args()
    args.seed > 0 && Random.seed!(args.seed)

    if args.cuda && has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
     
    # load data
    # train_data, val_data = DataUtils.get_train_validation(xtrain, ytrain, args.batch_size, args.train_prop, device)
    train_data, val_data = DataUtils.get_train_validation(DataUtils.data_prep(args.train_dir), readdlm(args.train_dir * "/y_train.txt", Int), args.batch_size, args.train_prop, device)

    # initialize model
    m = build_model(args.input_dims, args.nclasses)
    m = m |> device
    best_model = build_model(args.input_dims, args.nclasses)

    #optimizer
    opt = ADAM(args.η)

    ρ = cast_param(train_data.data[1], args.ρ)

    # parameters
    ps = Flux.params(m)

    # make path for model storage and log output
    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogging
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    #initialize tracking of accuracy and improvement
    best_acc = 0.0
    last_improvement = 0
    if device == gpu && CUDA.functional()
        augment(randn(Float32, 2, 4) |> gpu, 0.1f0, gpu)
    end

    # training
    train_steps = 0
    @info "Starting training. Total epochs: $(args.epochs)"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(train_data))

        training_function(m, train_data, ps, opt, ρ, device, progress)

        # calculate accuracy on validation set
        vacc = accuracy(val_data, m)
        @info "Validation set accuracy: $(vacc)"

        # log model and loss with TensorBoard
        if args.tblogging
            TBCallback(tblogger, train_data, val_data, m)
        end

        #If accuracy improves, save current model (so saved model is most performant)
        if vacc >= best_acc
            best_model = deepcopy(m)
            best_acc = vacc
            last_improvement = epoch
        end

        #If no improvement in args.lr_patience epochs, reduce learning rate
        if epoch - last_improvement >= args.lr_patience && opt.eta > 1e-6
            opt.eta /= args.γ
            @warn(" -> No improvement in $(args.lr_patience) epochs, reducing learning rate to $(opt.eta)!")
            # reset last_improvement to provide enough time to improve after reducing LR
            last_improvement = epoch
        end

        # Early stopping - Stop if no improvement in args.convergence epochs
        if epoch - last_improvement >= args.convergence
            @warn(" -> No improvement in $(args.convergence) epochs. Approximately converged.")
            break
        end
    end
    if args.save_model
        model_path = abspath(joinpath(args.save_path, "model.bson"))
        let model = best_model |> cpu, data = train_data |> cpu, args = args
            BSON.@save model_path model args data
            @info "Best model saved: $(model_path)"
        end
    end
    if device == gpu
        CUDA.reclaim()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    train(args)
end

end #module