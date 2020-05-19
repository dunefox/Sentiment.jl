module Ops
    export data, build_vocab, tokenise, load_emb_table, create_embeddings

    using Random, Embeddings

    function build_vocab(set)
        unique(Iterators.flatten(split.([s[1] for s in set], " ", keepempty=false)))
    end
    
    function data()
        folder = "./data/"
        train = []
        test = []
        for set in ["train", "test"]
            for label in ["pos", "neg"]
                # text[set][label] = []
                path = folder * "/" * set * "/" * label * "/"
                for file in readdir(path)
                    open(path * file) do f
                        line = strip(read(f, String), '\n')
                        line = tokenise(line)
                        if set == "train"
                            push!(train, (join(line, " "), label))
                        else
                            push!(test, (join(line, " "), label))
                        end
                    end
                end
            end
        end
        train, test
    end

    function tokenise(sentence)
        # Simplified assumption: disregard punctuation, etc.
        sentence = join([c for c in sentence if c ∉ "!?:;,.()[]\\<>/}{\"#%\$"])
        words = split(lowercase(sentence), r"[\s.]+", keepempty=false)
        
        pref, suf = r"^([´`'#(])*" => "", r"(['#(´`])*$" => ""
        words = replace.(words, pref)
        replace.(words, suf)
    end

    function f₁(preds, golds)
        TP, FP, FN = 0, 0, 0
        for (pred, gold) in zip(preds, golds)
            if pred == gold == 1
                TP += 1
            elseif pred == 1
                FP += 1
            elseif pred == 2
                FN += 1
            end
        end
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        2 * prec * rec / (prec + rec)
    end

    function create_embeddings(data, emb_table, get_word_index; number=500)
        samples = []
        for (xᵢ, yᵢ) in shuffle(data)[1:number]
            embs = reduce(hcat, [get_embedding(x, emb_table, get_word_index) for x in tokenise(xᵢ)])
            lbl = Dict("pos" => 1, "neg" => 2)[yᵢ]
            push!(samples, (embs, lbl))
        end
        samples
    end

    function get_embedding(word, emb_table, get_word_index)
        ind = get(get_word_index, word, get_word_index["unk"])
        emb = emb_table.embeddings[:,ind]
        return emb
    end

    function load_emb_table(path)
        @info("Loading embedding table...")
        embtable = Embeddings.load_embeddings(GloVe, path)
    end
end
