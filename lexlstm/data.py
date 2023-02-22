from src import batch_seqs
import torch


EPS = 1e-7


def encode(data, vocab_x, vocab_y):
    return [encode_io(datum, vocab_x, vocab_y) for datum in data]


def encode_io(datum, vocab_x, vocab_y):
    inp, out = datum
    inp = vocab_x.tokenize(inp)
    out = vocab_y.tokenize(out)
    return (
        [vocab_x.sos()] + vocab_x.encode(inp) + [vocab_x.eos()],
        [vocab_y.sos()] + vocab_y.encode(out) + [vocab_y.eos()],
    )


def eval_format(vocab, seq):
    if vocab.eos() in seq:
        seq = seq[: seq.index(vocab.eos()) + 1]
    seq = seq[1:-1]
    return vocab.decode(seq)


def eval_format_v2(tokenizer, seq):
    return tokenizer.decode(seq, skip_special_tokens=True)


def collate(batch, augmenter_fn=None):
    if augmenter_fn is not None:
        batch = list(map(augmenter_fn, batch))

    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    inp, out = zip(*batch)

    lens = torch.LongTensor(list(map(len, inp)))
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out, lens


def collate_v2(batch, *, tokenizer):
    inp_text, out_text = zip(*batch)

    inp = tokenizer.batch_encode_plus(
        list(inp_text),
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    out = tokenizer.batch_encode_plus(
        list(out_text),
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    inp_ids = inp["input_ids"]
    inp_mask = inp["attention_mask"]
    out_ids = out["input_ids"]
    out_mask = out["attention_mask"]

    return {
        "inp_ids": inp_ids.to(dtype=torch.long),
        "inp_mask": inp_mask.to(dtype=torch.long),
        "out_ids": out_ids.to(dtype=torch.long),
        "out_mask": out_mask.to(dtype=torch.long),
    }


def get_fig2_exp(input_symbols, output_symbols):
    study = [
        (["dax"], ["RED"]),
        (["lug"], ["BLUE"]),
        (["wif"], ["GREEN"]),
        (["zup"], ["YELLOW"]),
        (["lug", "fep"], ["BLUE", "BLUE", "BLUE"]),
        (["dax", "fep"], ["RED", "RED", "RED"]),
        (["lug", "blicket", "wif"], ["BLUE", "GREEN", "BLUE"]),
        (["wif", "blicket", "dax"], ["GREEN", "RED", "GREEN"]),
        (["lug", "kiki", "wif"], ["GREEN", "BLUE"]),
        (["dax", "kiki", "lug"], ["BLUE", "RED"]),
        (["lug", "fep", "kiki", "wif"], ["GREEN", "BLUE", "BLUE", "BLUE"]),
        (
            ["wif", "kiki", "dax", "blicket", "lug"],
            ["RED", "BLUE", "RED", "GREEN"],
        ),
        (["lug", "kiki", "wif", "fep"], ["GREEN", "GREEN", "GREEN", "BLUE"]),
        (
            ["wif", "blicket", "dax", "kiki", "lug"],
            ["BLUE", "GREEN", "RED", "GREEN"],
        ),
    ]

    test = [
        (["zup", "fep"], ["YELLOW", "YELLOW", "YELLOW"]),
        (["zup", "blicket", "lug"], ["YELLOW", "BLUE", "YELLOW"]),
        (["dax", "blicket", "zup"], ["RED", "YELLOW", "RED"]),
        (["zup", "kiki", "dax"], ["RED", "YELLOW"]),
        (["wif", "kiki", "zup"], ["YELLOW", "GREEN"]),
        (["zup", "fep", "kiki", "lug"], ["BLUE", "YELLOW", "YELLOW", "YELLOW"]),
        (
            ["wif", "kiki", "zup", "fep"],
            ["YELLOW", "YELLOW", "YELLOW", "GREEN"],
        ),
        (
            ["lug", "kiki", "wif", "blicket", "zup"],
            ["GREEN", "YELLOW", "GREEN", "BLUE"],
        ),
        (
            ["zup", "blicket", "wif", "kiki", "dax", "fep"],
            ["RED", "RED", "RED", "YELLOW", "GREEN", "YELLOW"],
        ),
        (
            ["zup", "blicket", "zup", "kiki", "zup", "fep"],
            ["YELLOW", "YELLOW", "YELLOW", "YELLOW", "YELLOW", "YELLOW"],
        ),
    ]
    return study, test
