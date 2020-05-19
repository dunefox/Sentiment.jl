module Models
    export Standard, Attention

    using Flux

    struct Standard
        dense1
        lstm
        dense2
    end

    Flux.@functor Standard

    function (m::Standard)(xᵢ)
        in  = m.dense1(xᵢ)
        hₙ   = m.lstm(in)[:, end]
        out = m.dense2(hₙ)
        softmax(out)
    end

    struct Attention
        dense1
        lstm
        attn
        v
        dense2
    end

    @Flux.functor Attention

    function (attn::Attention)(xᵢ)
        println("size xᵢ: ", size(xᵢ), typeof(xᵢ))
        in  = attn.dense1(xᵢ)
        println("size in: ", size(in), typeof(in))
        h   = attn.lstm(in)
        αₙ  = softmax(attn.v' * tanh.(attn.attn(h)), dims=2)
        println("size αₙ: ", size(αₙ), typeof(αₙ))
        println("size h: ", size(h), typeof(h))
        hₛ   = αₙ .* h
        println("size hₛ: ", size(hₛ), typeof(hₛ))
        out = attn.dense2(hₛ)
        softmax(out, dims=2)
    end
end
