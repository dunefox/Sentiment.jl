using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, params
using Base.Iterators: repeated, partition
using CuArrays

# using Pkg
# Pkg.add.(["Colors", "FileIO", "ImageShow", "Statistics"])

using Colors, FileIO, ImageShow

# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

# Partition into batches of size 1,000
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 1000)]

use_gpu = true # helper to easily switch between gpu/cpu

todevice(x) = use_gpu ? gpu(x) : x

train = todevice.(train)

# Prepare test set (first 1,000 images)
tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> todevice
tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9) |> todevice

m = Chain(
  Conv((2,2), 1=>16, relu),
  MaxPool((2, 2)),
  Conv((2,2), 16=>8, relu),
  MaxPool((2, 2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax
) |> todevice

loss(x, y) = crossentropy(m(x), y)

Flux.train!(loss, params(m), train, ADAM())

N = 7
img = tX[:, :, 1:1, N:N]
println("Predicted: ", Flux.onecold(m(gpu(img))) .- 1)

save("results/test.png", collect(tX[:, :, 1, N]))
