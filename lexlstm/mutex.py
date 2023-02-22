import collections

from data import eval_format
from src import EncDec
from src import RecordLoss
from src import Vocab
from src.t5_encdec import EncDecT5
from torch import nn


LossTrack = collections.namedtuple("LossTrack", "nll mlogpyx pointkl")


class Mutex(nn.Module):
    def __init__(
        self,
        vocab_x,
        vocab_y,
        emb,
        dim,
        copy=False,
        temp=1.0,
        max_len_x=8,
        max_len_y=8,
        n_layers=1,
        self_att=False,
        attention=True,
        dropout=0.0,
        bidirectional=True,
        rnntype=nn.LSTM,
        kl_lamda=1.0,
        recorder=RecordLoss(),
        qxy=None,
        embedding_file=None,
    ):
        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.dim = dim
        self.n_layers = n_layers
        self.temp = temp
        self.MAXLEN_X = max_len_x
        self.MAXLEN_Y = max_len_y

        if isinstance(vocab_x, Vocab):
            EncDecInit = EncDec
        else:
            EncDecInit = EncDecT5

        self.pyx = EncDecInit(
            vocab_x,
            vocab_y,
            emb,
            dim,
            copy=copy,
            n_layers=n_layers,
            self_att=self_att,
            source_att=attention,
            dropout=dropout,
            bidirectional=bidirectional,
            rnntype=rnntype,
            MAXLEN=self.MAXLEN_Y,
            embedding_file=embedding_file,
        )
        if qxy:
            self.qxy = EncDecInit(
                vocab_y,
                vocab_x,
                emb,
                dim,
                copy=copy,
                n_layers=n_layers,
                self_att=self_att,
                dropout=dropout,
                bidirectional=bidirectional,
                rnntype=rnntype,
                source_att=attention,
                MAXLEN=self.MAXLEN_X,
                embedding_file=embedding_file,
            )
            # self.qxy = None
        self.recorder = recorder

    def forward(self, inp, out, lens=None, recorder=None):
        return self.pyx(inp, out, lens=lens)

    def print_tokens(self, vocab, tokens):
        return [" ".join(eval_format(vocab, tokens[i])) for i in range(len(tokens))]

    def sample(self, *args, **kwargs):
        return self.pyx.sample(*args, **kwargs)
