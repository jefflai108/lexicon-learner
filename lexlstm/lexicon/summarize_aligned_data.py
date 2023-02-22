from collections import Counter
import json
import sys


def main(data_file, aligner_file, SPLIT_TOK):
    data = None
    with open(data_file, "r") as f:
        data = f.read().splitlines()

    word_adjacency = {}
    with open(aligner_file, "r") as f:
        for k, line in enumerate(f):
            input, output, *_ = data[k].split(SPLIT_TOK)
            input, output = input.strip().split(" "), output.strip().split(" ")
            alignments = line.strip().split(" ")
            for a in alignments:
                if len(a) == 0:
                    continue
                i, j = a.split("-")
                try:
                    wi = input[int(i)]
                except:
                    print("input:", input)
                    print("i:", i)
                    print("a:", a)

                try:
                    wj = output[int(j)]
                except:
                    print("output:", output)
                    print("len(output)", len(output))
                    print("j:", j)
                    print("a:", a)
                if wi in word_adjacency:
                    word_adjacency[wi].append(wj)
                else:
                    word_adjacency[wi] = [wj]

    word_alignment = {}
    for k, v in word_adjacency.items():
        if len(v) == 0:
            word_alignment[k] = {k: 0}
        else:
            word_alignment[k] = dict(Counter(v))

    # with open(aligner_file + '.pickle', 'wb') as handle:
    #     pickle.dump(word_alignment, handle)
    with open(aligner_file + ".json", "w") as handle:
        json.dump(word_alignment, handle)


if __name__ == "__main__":
    # execute only if run as a script
    assert len(sys.argv) > 2
    if len(sys.argv) == 4:
        SPLIT_TOK = sys.argv[3]
    else:
        SPLIT_TOK = " ||| "

    print("SPLIT TOKEN:", SPLIT_TOK)

    main(sys.argv[1], sys.argv[2], SPLIT_TOK)
