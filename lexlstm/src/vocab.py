import json


class Vocab(object):
    PAD = "<pad>"
    SOS = "<s>"
    EOS = "</s>"
    COPY = "<copy>"
    UNK = "<unk>"

    tokenization_character = " "

    def __init__(self):
        self.contents = {}
        self.rev_contents = {}
        self.add(self.PAD)
        self.add(self.SOS)
        self.add(self.EOS)
        self.add(self.COPY)
        self.add(self.UNK)

    def add(self, sym):
        if sym not in self.contents:
            i = len(self.contents)
            self.contents[sym] = i
            self.rev_contents[i] = sym
        return self

    def merge(self, add_vocab):
        for sym in add_vocab.contents.keys():
            self.add(sym)
        return self

    def __getitem__(self, sym):
        return self.contents[sym]

    def __contains__(self, sym):
        return sym in self.contents

    def __len__(self):
        return len(self.contents)

    def encode(self, seq, unk=True):
        if unk:
            seq = [s if s in self else self.UNK for s in seq]
        return [self[i] for i in seq]

    def decode(self, seq):
        return [self.rev_contents[i] for i in seq]

    def get(self, i):
        return self.rev_contents[i]

    def pad(self):
        return self.contents[self.PAD]

    def sos(self):
        return self.contents[self.SOS]

    def eos(self):
        return self.contents[self.EOS]

    def copy(self):
        return self.contents[self.COPY]

    def unk(self):
        return self.contents[self.UNK]

    def tokenize(self, x):
        if type(x) == str:
            return x.split(self.tokenization_character)
        else:
            return list(map(self.tokenize, x))

    def __str__(self):
        out = (
            ["Vocab("] + ["\t%s:\t%s" % pair for pair in self.contents.items()] + [")"]
        )
        return "\n".join(out)

    def dump(self, writer):
        json.dump(self.contents, writer)

    def load(self, reader):
        newcontents = json.load(reader)
        for k, v in newcontents.items():
            if k in self.contents:
                assert self.contents[k] == v
            self.contents[k] = v
            self.rev_contents[v] = k
