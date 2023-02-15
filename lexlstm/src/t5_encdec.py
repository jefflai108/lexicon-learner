from warnings import WarningMessage

from torch import nn
from transformers import AutoModelWithLMHead


class EncDecT5(nn.Module):
    def __init__(
        self,
        vocab_x,
        vocab_y,
        emb,
        dim,
        model_name="t5-base",
        copy=False,
        n_layers=1,
        self_att=False,
        dropout=0.0,
        bidirectional=True,
        rnntype=nn.LSTM,
        MAXLEN=45,
        source_att=False,
        embedding_file=None,
    ):
        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.tokenizer = vocab_x
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.nll = nn.CrossEntropyLoss(reduction="sum")  # TODO: why mean better?
        self.nll_wr = nn.CrossEntropyLoss(reduction="none")
        self.dim = dim
        self.n_layers = n_layers
        self.MAXLEN = MAXLEN
        self.source_att = source_att
        self.self_att = self_att
        self.concat_feed = self_att or source_att
        self.model_name = model_name

        if self.bidirectional:
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.proj = nn.Identity()

        self.model = AutoModelWithLMHead.from_pretrained(model_name)

        self.encoder = self.model.encoder

        self.decoder = self.model.decoder

        if embedding_file is not None:
            raise WarningMessage("Embedding file not implemented yet for T5")
            pass

    def forward(self, inp, out, *, inp_mask, out_mask, per_instance=False):
        # Run T5 model with the data
        output = self.model(input_ids=inp, attention_mask=inp_mask, labels=out)
        return output.loss

    def sample(
        self,
        inp,
        max_len,
        inp_mask=None,
        lens=None,
        prompt=None,
        greedy=False,
        top_p=None,
        temp=1.0,
        custom_sampler=None,
        beam_size=1,
        calc_score=False,
        **kwargs,
    ):
        outputs = self.model.generate(
            input_ids=inp,
            attention_mask=inp_mask,
            max_length=max_len,
            do_sample=False,  # disable sampling to test if batching affects output
        )

        return outputs
