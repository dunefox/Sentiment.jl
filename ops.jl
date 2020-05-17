module Ops
    export data, build_vocab, tokenise

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
            if pred == gold == "pos"
                TP += 1
            elseif pred == "pos"
                FP += 1
            elseif pred == "neg"
                FN += 1
            end
        end
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        2 * prec * rec / (prec + rec)
    end
end
