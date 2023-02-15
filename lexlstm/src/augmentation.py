import json
import random

from data import collate_v2
from data import encode_io
from mutex import Vocab


def read_augmented_file(file, vocab_x, vocab_y):
    with open(file, "r") as f:
        data = json.load(f)
    edata = []
    for datum in data:
        inp, out = datum["inp"], datum["out"]
        edata.append(encode_io((inp, out), vocab_x, vocab_y))
    return edata


def swap_ids(tensor, id1, id2, substitute=False):
    if substitute:
        tensor.masked_fill_(tensor == id1, id2)
    else:
        tensor.masked_fill_(tensor == id1, -1)
        tensor.masked_fill_(tensor == id2, id1)
        tensor.masked_fill_(tensor == -1, id2)


def swap_tokens(text, token1, token2, substitute=False):
    if substitute:
        text = text.replace(token1, token2)
    else:
        text = text.replace(token1, "<special>")
        text = text.replace(token2, token1)
        text = text.replace("<special>", token2)
    return text


def make_a_swap_batch(
    inputs,
    outputs,
    vocab_x: Vocab,
    vocab_y: Vocab,
    lexicon,
    swapables,
    steps=0,
    substitute=False,
):
    for i in range(inputs.shape[-1]):
        inp = inputs.select(-1, i)
        out = outputs.select(-1, i)
        make_a_swap_single(
            inp,
            out,
            vocab_x,
            vocab_y,
            lexicon,
            swapables,
            steps=steps,
            substitute=substitute,
        )


def make_a_swap_batch_t5(
    inputs,
    outputs,
    vocab_x: Vocab,
    vocab_y: Vocab,
    lexicon,
    swapables,
    steps=0,
    substitute=False,
):
    batch = []
    for i in range(inputs.shape[0]):
        inp = inputs.select(0, i)
        out = outputs.select(0, i)
        inp_decoded, out_decoded = make_a_swap_single_t5(
            inp,
            out,
            vocab_x,
            vocab_y,
            lexicon,
            swapables,
            steps=steps,
            substitute=substitute,
        )
        batch.append((inp_decoded, out_decoded))
    return collate_v2(batch, tokenizer=vocab_x)


def make_a_swap_single_t5(
    inp,
    out,
    vocab_x: Vocab,
    vocab_y: Vocab,
    lexicon,
    swapables,
    steps=0,
    substitute=False,
):
    # inp_decoded = vocab_x.decode(inp)
    # out_decoded = vocab_y.decode(out)
    inp_decoded = inp
    out_decoded = out

    keys = list(filter(lambda k: k in inp_decoded, lexicon.keys()))

    ## Add substitute

    if len(keys) != 0:
        k1 = random.choice(keys)
        weights = [1 / next(iter(lexicon[k].values())) for k in swapables[k1]]
        k2 = random.choices(swapables[k1], weights=weights, k=1)[0]
        ks = [k1, k2]
    else:
        k1 = random.choice(list(lexicon.keys()))
        weights = [1 / next(iter(lexicon[k].values())) for k in swapables[k1]]
        k2 = random.choices(swapables[k1], weights=weights, k=1)[0]
        ks = [k1, k2]

    # if steps == 0:
    #     print(ks)

    inp_decoded = swap_tokens(inp_decoded, *ks, substitute=substitute)

    if substitute:
        for v, _ in lexicon[ks[0]].items():
            code2 = random.choice(list(lexicon[ks[1]].keys()))
            out_decoded = out_decoded.replace(v, code2)
    else:
        for v, _ in lexicon[ks[0]].items():
            out_decoded = out_decoded.replace(v, "@")

        for v, _ in lexicon[ks[1]].items():
            code1 = random.choice(list(lexicon[ks[0]].keys()))
            out_decoded = out_decoded.replace(v, code1)

        code2 = random.choice(list(lexicon[ks[1]].keys()))

        out_decoded = out_decoded.replace("@", code2)

    return inp_decoded, out_decoded


def make_a_swap_single(
    inp,
    out,
    vocab_x: Vocab,
    vocab_y: Vocab,
    lexicon,
    swapables,
    steps=0,
    substitute=False,
):
    keys = list(filter(lambda k: vocab_x.contents[k] in inp, lexicon.keys()))

    ## Add substitute

    if len(keys) != 0:
        k1 = random.choice(keys)
        weights = [1 / next(iter(lexicon[k].values())) for k in swapables[k1]]
        k2 = random.choices(swapables[k1], weights=weights, k=1)[0]
        ks = [k1, k2]
    else:
        k1 = random.choice(list(lexicon.keys()))
        weights = [1 / next(iter(lexicon[k].values())) for k in swapables[k1]]
        k2 = random.choices(swapables[k1], weights=weights, k=1)[0]
        ks = [k1, k2]

    if steps == 0:
        print(ks)

    ks_q_id = [vocab_x.contents[k] for k in ks]
    swap_ids(inp, *ks_q_id, substitute=substitute)

    if substitute:
        for v, _ in lexicon[ks[0]].items():
            code2 = random.choice(list(lexicon[ks[1]].keys()))
            out.masked_fill_(out == vocab_y[v], vocab_y[code2])
    else:
        for v, _ in lexicon[ks[0]].items():
            out.masked_fill_(out == vocab_y[v], -1)

        for v, _ in lexicon[ks[1]].items():
            code1 = random.choice(list(lexicon[ks[0]].keys()))
            out.masked_fill_(out == vocab_y[v], vocab_y[code1])

        code2 = random.choice(list(lexicon[ks[1]].keys()))

        out.masked_fill_(out == -1, vocab_y[code2])
