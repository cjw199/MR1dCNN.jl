# Multi-Resolution One-Dimensional Convolutional Neural Network
## Overview

This example demonstrates the use of a 1-dimensional multi-resolution convolutional neural network (CNN) to classify human activity from time-resolved data using Julia and the Flux machine learning library.  The data are from the Human Activity Recognition (HAR) Using Smartphone Dataset from the study described in the paper:
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012

Data were collected on an experimental group performing six activities (walking, walking upstairs,walking downstairs, sitting, standing, and laying) while wearing smartphones.  The captured sensor signals were used to create a dataset of tri-axial linear acceleration and tri-axial angular velocity measurements.  The processed dataset is available from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), although it is provided with this example for convenience.

## Data

The dataset consists of a training and a testing set.  The training set contains 7,352 examples of the 9 features (3-axis linear acceleration, 3-axis estimated body acceleration, and 3-axis angular velocity) in 128-time step sequences.  Each feature is stored as an individual file, so the dataset consists of 9 files.  This example includes tools to aggregate the data into the tensor format needed by the CNN model.

## Motivation

The motivating idea behind using a CNN on sequence data is analogous to its implementation in the context of image recognition.  The convolution operation can abstract away from the raw input and express the data in feature maps that richly represent the activity types.  These feature maps can greatly enhance the representational power of the model, where each feature map can be thought of as a unique summarization of the data.  By setting the kernel dimensions to span all features with no vertical padding, the convolution operation is constrained to operate along the temporal axis, thereby capturing the sequential and dynamic structure of the data.

Yet, a multivariate time series dataset can exhibit patterns at multiple levels of temporal resolution.  A single initial kernel pass over the series may not capture all of the inherent temporal structure of the data.  Therefore, this approach combines the outputs of multiple CNNs that convolve over the time dimension with increasing levels of temporal abstraction.  That is to say, the most granular model uses relatively small kernels and strides, whereas the most general uses relatively large kernels and strides.  The model uses an aggregate loss function on the mean cross entropy of the component model outputs.

## Methodological Notes

In this example, the data are structured as a 4th-order tensor of ***features*** x ***time steps*** x ***channels*** x ***examples***, so each input example is of dimension **9** x **128** x **1**. The kernels convolve over the time dimension and produce feature maps along the channel dimension.  Data augmentation is used as a regularizer, by means of random Gaussian noise.

Train/test split and various other parameters can be adjusted via an arguments struct.

Training data can be logged using TensorBoardLogger to facilitate diagnostics using Tensorboard.  This feature is off by default and can be changed in the arguments struct.

## Usage

### Setup

From the project directory, run

`julia -e "using Pkg; Pkg.activate(\".\"); Pkg.build();"`

 to activate the project and install all dependencies.

### Running the demo

#### Quickstart

From the REPL, load the project.

`include("src/MR1dCNN.jl")`

`using .MR1dCNN`

For a quick start with default parameters and model architecture, the demo itself can be run by 

`train(args)`.

This training function takes one argument: the training parameters object (here, `args`). 
The data will be automatically loaded and transformed, training executed, and (provided `args.save_model` remains set to `true`) save the best-performing model to an output directory.  It will also (if `args.tblogging` is set to `true`) log model parameters and performance data on a per-epoch basis to a Tensorboard log file.

#### Basic Usage

Arguments are in an `Args` struct defined in `MR1dCNN.jl`.  When loaded, the `args` object is created and arguments can be adjusted.  For example, to change the number of epochs from 10 (the default) to 20, use

`args.epochs = 20`.

The model architecture is defined in ```arch/arch.json```.  This file contains a definition for each model in Flux syntax as a string.  To change or use your own model architectures, just edit the ```arch.json``` file with the appropriate changes.  It is advised to ensure that any changes will work on the data (e.g., that layer parameters are compatible with the shape of the data).

#### TensoBoard Logging

The logged data can be visualized by running the following command (assuming you have installed Tensorboard via pip):
```
tensorboard --logdir [path to output directory]

#example

tensorboard --logdir output1
```
The resulting output in your web browser will be similar to this.

![img1](img/tensorboard_img.PNG)
![img2](img/tnsorboard1_img2.PNG)

Testing data is provided in the data directory, and the saved model can be used to run the test.

## Results

To test on the included testing set, run

 ```
using Flux, Random, StatsBase

MR1dCNN.test("<path to saved model>")
```
The test function takes the ```scale``` argument to use the saved scaling transform to scale the testing data. Please ensure that if you chose not to scale data during training, this parameter must be set tp ```false```.

You should be able to achieve at or above 95% training accuracy in 10 epochs, and over 90% test accuracy using the demo code and default arguments.
