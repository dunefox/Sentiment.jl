using Flux, CSV, StatsBase, Query, Random, ProgressMeter
include("ops.jl")
include("models.jl")

train, test = Ops.data();
vocab = Ops.build_vocab(train)

# Turn off bracket autocompletion in the REPL temporarily
# OhMyREPL.enable_autocomplete_brackets(false)

classes = "pos", "neg"
labels = 1:2
emb_dim = 300
hidden_dim = 50
attn_dim = 30
@info("Parameters ", classes, labels, emb_dim, hidden_dim, attn_dim)

# model = Models.Standard(
#     Dense(emb_dim, hidden_dim, σ),
#     LSTM(hidden_dim, hidden_dim),
#     Dense(hidden_dim, length(labels))
# )
# @info("Model -> Standard", model)

model = Models.Attention(
    Dense(emb_dim, hidden_dim),
    LSTM(hidden_dim, hidden_dim),
    Dense(hidden_dim, attn_dim),
    Dense(hidden_dim, attn_dim),
    rand(attn_dim),
    Dense(hidden_dim, length(labels))
)
@info("Model -> Attention", model)

# path = "/home/paul/projekte/julia/jposat/glove/glove.840B.300d.txt"
path = "/big/f/fuchsp/posat-adapted/glove/glove.840B.300d.txt"

const emb_table = Ops.load_emb_table(path)
const word_index = Dict(word=>ii for (ii,word) in enumerate(emb_table.vocab))

opt = ADAM()
loss(xᵢ, yᵢ) = -log(sum(model(xᵢ) .* Flux.onehot(yᵢ, labels)))
ps = params(model)

@info("Creating train set...")
tr_samples = Ops.create_embeddings(train, emb_table, word_index) 

Flux.testmode!(model, false)

@info("Beginning training...")
for epochᵢ in 1:6
    @info("Epoch $(epochᵢ) start")
    batch_loss = 0.0

    @showprogress 5 "Epoch $(epochᵢ): " for (i, (xᵢ, yᵢ)) in enumerate(tr_samples)
        gs = gradient(ps) do
            training_loss = loss(xᵢ, yᵢ)
            batch_loss += training_loss
            # reg = sum(params(model)[1] .^ 2)
            # Insert what ever code you want here that needs Training loss, e.g. logging
            training_loss
        end
        # insert what ever code you want here that needs gradient
        # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
        Flux.update!(opt, ps, gs)
        # Here you might like to check validation set accuracy, and break out to do early stopping
        Flux.reset!(model)
    end
    @info("Epoch $(epochᵢ) end", batch_loss)
end

Flux.testmode!(model, true)

@info("Creating test set...")
te_samples = Ops.create_embeddings(test, emb_table, word_index, number=500)

@info("Calculating F₁-Score...")
preds, gold = [], []
for (xᵢ, yᵢ) in te_samples
    Flux.reset!(model)
    # println("##############")
    # println("size xᵢ: ", size(xᵢ))
    # println("Model output: ", model(xᵢ))
    # println("##############")
    push!(preds, Flux.onecold(model(xᵢ), labels))
    push!(gold, yᵢ)
end

@info("F₁: $(Ops.f₁(preds, gold))")

