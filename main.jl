using Flux, CSV, StatsBase, Query, Embeddings, Random, CuArrays
include("ops.jl")

train, test = Ops.data();
vocab = Ops.build_vocab(train)

# Turn off bracket autocompletion in the REPL temporarily
OhMyREPL.enable_autocomplete_brackets(false)

classes = "pos", "neg"
labels = 1:2
emb_dim = 300
hidden_dim = 50

module M
    using Flux

    classes = "pos", "neg"
    labels = 1:2
    emb_dim = 300
    hidden_dim = 50

    struct Model
        dense1
        lstm
        dense2
        
        Model(
            dense1  = Dense(emb_dim, hidden_dim, σ),
            lstm    = LSTM(hidden_dim, hidden_dim),
            dense2  = Dense(hidden_dim, 2)
        ) = new(dense1, lstm, dense2)
    end

    Flux.@functor Model
    
    function (m::Model)(xᵢ)
        in = m.dense1.(xᵢ)
        h = m.lstm.(in)[end]
        # attention layer...
        out = m.dense2(h)
        softmax(out)
    end
end

model = M.Model()

const embtable = Embeddings.load_embeddings(GloVe, "/home/paul/projekte/julia/jposat/glove/glove.840B.300d.txt")
const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

function get_embedding(word)
    ind = get(get_word_index, word, get_word_index["unk"])
    emb = embtable.embeddings[:,ind]
    return emb
end

opt = ADAM()
loss(xᵢ, yᵢ) = -log(sum(model(xᵢ) .* Flux.onehot(Dict("pos" => 1, "neg" => 2)[yᵢ], labels)))
ps = params(model)

# samples = [get_embedding(x) for x in Ops.tokenise(xᵢ) for xᵢ in shuffle(train[1:200])]

tr_samples = []
for (xᵢ, yᵢ) in shuffle(train)[1:500]
    push!(tr_samples, ([get_embedding(x) for x in Ops.tokenise(xᵢ)], yᵢ))
end

Flux.testmode!(model, false)

for epochᵢ in 1:5
    @info("Epoch $(epochᵢ)")
    batch_loss = 0.0

    for (xᵢ, yᵢ) in tr_samples
        Flux.reset!(model)

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
    end
    @info("Loss: $(batch_loss)")
end

Flux.testmode!(model, true)

te_samples = []
for (xᵢ, yᵢ) in shuffle(test)[1:200]
    push!(te_samples, ([get_embedding(x) for x in Ops.tokenise(xᵢ)], yᵢ))
end

preds, gold = [], []
for (xᵢ, yᵢ) in te_samples
    push!(preds, Flux.onecold(model(xᵢ), classes))
    push!(gold, yᵢ)
end

@info("F₁: $(Ops.f₁(preds, gold))")
