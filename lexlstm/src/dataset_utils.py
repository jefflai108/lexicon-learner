import abc
from collections import Counter
import math
import os
import random
import re
from typing import Any, List

from data import encode
from data import encode_io
from data import get_fig2_exp
import hlog
import numpy as np
from src import augmentation
from src import batch_seqs
from src import Vocab
from src.projection import SoftAlign
import torch
from torch.utils.data import Dataset


class Seq2SeqDataset(abc.ABC, Dataset):
    """Base class for all sequence to sequence datasets used in this repo."""

    data: List
    max_len_x: int
    max_len_y: int
    augmentation_probability: float = 0.0
    vocab_x: Any
    vocab_y: Any
    lexicon: Any
    swapables: Any
    substitute: bool = False

    @staticmethod
    @abc.abstractmethod
    def load_dataset(
        root: str = "SCAN",
        vocab_x=None,
        vocab_y=None,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def copy_translation(self, use_lexicon=False):
        pass

    def augmenter(self, datum):
        return datum

    def eval_format(self, seq, vocab):
        if isinstance(vocab, Vocab):
            if vocab.eos() in seq:
                seq = seq[: seq.index(vocab.eos()) + 1]
            seq = seq[1:-1]
            return vocab.decode(seq)
        else:
            return vocab.decode(seq, skip_special_tokens=True)

    def enable_augmentation(self, ratio=0.2):
        self.augmentation_probability = ratio

    def disable_augmentation(self):
        self.augmentation_probability = 0

    def preprocess(self, datum):
        return datum

    def collate_fn(self, batch):
        if (
            self.augmentation_probability > 0
            and random.random() < self.augmentation_probability
        ):
            batch = [self.augmenter(datum) for datum in batch]

        batch = [self.preprocess(datum) for datum in batch]

        if isinstance(self.vocab_x, Vocab):
            batch = encode(batch, self.vocab_x, self.vocab_y)
            batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
            inputs, targets = zip(*batch)
            lengths = torch.LongTensor(list(map(len, inputs)))
            inputs = batch_seqs(inputs)
            targets = batch_seqs(targets)

            return {"inputs": inputs, "targets": targets, "lengths": lengths}
        else:
            inp_text, out_text = zip(*batch)

            inp = self.vocab_x.batch_encode_plus(
                list(inp_text),
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            out = self.vocab_y.batch_encode_plus(
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
                "inputs": inp_ids.to(dtype=torch.long),
                "inputs_mask": inp_mask.to(dtype=torch.long),
                "targets": out_ids.to(dtype=torch.long),
                "targets_mask": out_mask.to(dtype=torch.long),
            }


class SCANDataset(Seq2SeqDataset):
    """SCAN dataset from Brandon and Lake."""

    raw_data_reg_exp: str = re.compile("^IN\:\s(.*?)\sOUT\: (.*?)$")

    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute=False,
        lexicon=None,
        swapables=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.swapables = swapables
        self.substitute = substitute
        self.references = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def augmenter(self, datum):
        inputs, targets = datum["inputs"], datum["targets"]
        inputs, targets = augmentation.make_a_swap_single_t5(
            inputs,
            targets,
            self.vocab_x,
            self.vocab_y,
            self.lexicon,
            self.swapables,
            substitute=self.substitute,
        )
        return {"inputs": inputs, "targets": targets}

    def preprocess(self, datum):
        return datum["inputs"], datum["targets"]

    @staticmethod
    def load_dataset(
        root: str = "/afs/csail.mit.edu/u/a/akyurek/akyurek/git/align/SCAN/",
        vocab_x=None,
        vocab_y=None,
        subset: str = "around_right",
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        if vocab_x is None:
            vocab_x = Vocab()

        if vocab_y is None:
            vocab_y = Vocab()

        max_len_x, max_len_y = 0, 0

        data = {}

        for split in ("train", "test"):
            if subset == "around_right":
                path = os.path.join(
                    root,
                    f"template_split/tasks_{split}_template_around_right.txt",
                )
            else:
                path = os.path.join(
                    root, f"add_prim_split/tasks_{split}_addprim_jump.txt"
                )

            split_data = []
            with open(path, "r") as handle:
                for line in handle:
                    m = SCANDataset.raw_data_reg_exp.match(line)

                    str_inp, str_out = m.groups(1)

                    inp, out = (
                        vocab_x.tokenize(str_inp),
                        vocab_y.tokenize(str_out),
                    )

                    max_len_x = max(len(inp), max_len_x)
                    max_len_y = max(len(out), max_len_y)

                    if isinstance(vocab_x, Vocab):
                        for t in inp:
                            vocab_x.add(t)

                        for t in out:
                            vocab_y.add(t)

                    split_data.append({"inputs": str_inp, "targets": str_out})

            data[split] = split_data

        val_size = math.floor(len(data["train"]) * 0.01)
        train_size = len(data["train"]) - val_size

        train_items, val_items = torch.utils.data.random_split(
            data["train"], [train_size, val_size]
        )

        test_items = data["test"]

        max_len_x += 1
        max_len_y += 1

        hlog.value("max_len_x: ", max_len_x)
        hlog.value("max_len_y: ", max_len_y)
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])

        datasets = {
            "train": SCANDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": SCANDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": SCANDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }

        return datasets, vocab_x, vocab_y

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            proj = np.identity(len(self.vocab_x))
            vocab_keys = list(self.vocab_x.contents.keys())
            for x, y in [
                ("jump", "I_JUMP"),
                ("walk", "I_WALK"),
                ("look", "I_LOOK"),
                ("run", "I_RUN"),
                ("right", "I_TURN_RIGHT"),
                ("left", "I_TURN_LEFT"),
            ]:
                idx = vocab_keys.index(x)
                idy = vocab_keys.index(y)
                print("x: ", x, " y: ", y, "idx: ", idx, "idy: ", idy)
                proj[idx, idx] = 0
                proj[idx, idy] = 1
            return np.argmax(proj, axis=1)


class TextAlchemyDataset(Seq2SeqDataset):
    """TextAlchemy dataset from SCONE dataset"""

    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.substitute = substitute
        self.swapables = swapables
        self.references = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def augmenter(self, datum):
        instructions, states = datum["instructions"], datum["states"]
        instructions, states = augmentation.make_a_swap_single_t5(
            instructions,
            states,
            self.vocab_x,
            self.vocab_y,
            self.lexicon,
            self.swapables,
            substitute=self.substitute,
        )
        return {"instructions": instructions, "states": states}

    def preprocess(self, datum):
        instructions, states = datum["instructions"], datum["states"]
        instructions = instructions.split(" . ")
        states = states.split(" . ")
        assert len(instructions) + 1 == len(states), (
            len(instructions),
            len(states),
        )
        # input = [states[t] + ' | ' + instructions[t] for t in range(len(instructions))]
        input = states[0] + " | " + " . ".join(instructions)
        output = states[-1]
        # print("Input: ", ' '.join(input))
        # print("Output: ", ' '.join(output))
        return (input, output)

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            hlog.log("No lexicon provided using identity")
            proj = np.zeros((len(self.vocab_x), len(self.vocab_y)), dtype=np.float32)
            x_keys = list(self.vocab_x.contents.keys())
            y_keys = list(self.vocab_y.contents.keys())
            for x, x_key in enumerate(x_keys):
                if x_key in y_keys:
                    y = y_keys.index(x_key)
                    proj[x, y] = 1.0

            return np.argmax(proj, axis=1)

    @staticmethod
    def load_dataset(
        root: str = "/raid/lingo/akyurek/git/lexgen/data/rlong/",
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        data = {}

        if vocab_x is None:
            vocab_x = Vocab()

        if vocab_y is None:
            vocab_y = Vocab()

        max_len_x = 0
        max_len_y = 0

        for split in ("train", "dev", "test"):
            split_data = []
            path = os.path.join(root, f"alchemy-{split}.tsv.processed.tokenized")

            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    instructions_text, states_text, *_ = line.split(" ||| ")
                    instructions_text = instructions_text.strip()
                    states_text = states_text.strip()
                    instructions, states = (
                        instructions_text.split(" . "),
                        states_text.split(" . "),
                    )
                    assert len(instructions) + 1 == len(states), (
                        len(instructions),
                        len(states),
                    )

                    for time in range(len(instructions)):
                        str_inp = (
                            states[0] + " | " + " . ".join(instructions[: time + 1])
                        )
                        str_out = states[time + 1]

                        inp = vocab_x.tokenize(str_inp)
                        out = vocab_y.tokenize(str_out)

                        if len(split_data) < 10:
                            print(f"Input: {inp}\nOutput: {out}")

                        max_len_x = max(len(inp), max_len_x)
                        max_len_y = max(len(out), max_len_y)

                        if isinstance(vocab_x, Vocab):
                            for t in inp:
                                vocab_x.add(t)
                                vocab_y.add(t)

                            for t in out:
                                vocab_y.add(t)
                                vocab_x.add(t)

                        split_data.append(
                            {
                                "instructions": " . ".join(instructions[: time + 1]),
                                "states": " . ".join(states[: time + 2]),
                            }
                        )
            data[split] = split_data
        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])
        train_items = data["train"]
        val_items = data["dev"]
        test_items = data["test"]

        datasets = {
            "train": TextAlchemyDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": TextAlchemyDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": TextAlchemyDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }

        return datasets, vocab_x, vocab_y


class COGSDataset(Seq2SeqDataset):
    """COGS dataset from Kim et al. 2019"""

    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.substitute = substitute
        self.swapables = swapables
        self.references = None
        self.augmentation_probability = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def augmenter(self, datum):
        inputs, targets = datum["inputs"], datum["targets"]
        inputs, targets = augmentation.make_a_swap_single_t5(
            inputs,
            targets,
            self.vocab_x,
            self.vocab_y,
            self.lexicon,
            self.swapables,
            substitute=self.substitute,
        )
        return {"inputs": inputs, "targets": targets}

    def preprocess(self, datum):
        return datum["inputs"], datum["targets"]

    @staticmethod
    def load_dataset(
        root: str = "/raid/lingo/akyurek/git/align/COGS/cogs",
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        data = {}

        if vocab_x is None:
            vocab_x = Vocab()

        if vocab_y is None:
            vocab_y = Vocab()

        max_len_x = 0
        max_len_y = 0

        for split in ("train", "dev", "test", "gen"):
            split_data = []

            with open(f"{root}/{split}.tsv", "r", encoding="utf-8") as handle:
                for line in handle:
                    str_text, str_sparse, *_ = line.split("\t")
                    inp = vocab_x.tokenize(str_text)
                    out = vocab_y.tokenize(str_sparse)
                    max_len_x = max(len(inp), max_len_x)
                    max_len_y = max(len(out), max_len_y)
                    if isinstance(vocab_x, Vocab):
                        for t in inp:
                            vocab_x.add(t)
                            vocab_y.add(t)
                        for t in out:
                            vocab_y.add(t)
                            vocab_x.add(t)

                    split_data.append({"inputs": str_text, "targets": str_sparse})

            data[split] = split_data
        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])

        train_items = data["train"]
        val_items = data["test"]
        test_items = data["gen"]

        datasets = {
            "train": COGSDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": COGSDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": COGSDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }

        return datasets, vocab_x, vocab_y

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            hlog.log("No lexicon provided using identity")
            proj = np.zeros((len(self.vocab_x), len(self.vocab_y)), dtype=np.float32)
            x_keys = list(self.vocab_x.contents.keys())
            y_keys = list(self.vocab_y.contents.keys())
            for x, x_key in enumerate(x_keys):
                if x_key in y_keys:
                    y = y_keys.index(x_key)
                    proj[x, y] = 1.0

            return np.argmax(proj, axis=1)


class SPEECHDataset(Seq2SeqDataset):
    """SPEECH Dataset"""

    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.substitute = substitute
        self.swapables = swapables
        self.references = None
        self.augmentation_probability = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def augmenter(self, datum):
        inputs, targets = datum["inputs"], datum["targets"]
        inputs, targets = augmentation.make_a_swap_single_t5(
            inputs,
            targets,
            self.vocab_x,
            self.vocab_y,
            self.lexicon,
            self.swapables,
            substitute=self.substitute,
        )
        return {"inputs": inputs, "targets": targets}

    def preprocess(self, datum):
        return datum["inputs"], datum["targets"]

    @staticmethod
    def load_dataset(
        root: str = "/raid/lingo/akyurek/git/align/SPEECH/",
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        data = {}

        if vocab_x is None:
            vocab_x = Vocab()

        if vocab_y is None:
            vocab_y = Vocab()

        max_len_x = 0
        max_len_y = 0

        for split in ("train", "dev", "test"):
            split_data = []

            if split == "train":
                fname = "train_mined_t1.09_ekin.filter1000.tsv"
            else:
                fname = "valid_vp_ekin.filter1000.tsv"

            with open(f"{root}/{fname}", "r", encoding="utf-8") as handle:
                for line in handle:
                    src, tgt = line.split("\t")
                    inp = vocab_x.tokenize(src)
                    out = vocab_y.tokenize(tgt)
                    max_len_x = max(len(inp), max_len_x)
                    max_len_y = max(len(out), max_len_y)
                    if isinstance(vocab_x, Vocab):
                        for t in inp:
                            vocab_x.add(t)
                            vocab_y.add(t)
                        for t in out:
                            vocab_y.add(t)
                            vocab_x.add(t)

                    split_data.append({"inputs": src, "targets": tgt})

            data[split] = split_data
        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])

        train_items = data["train"]
        val_items = data["dev"]
        test_items = data["test"]

        datasets = {
            "train": SPEECHDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": SPEECHDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": SPEECHDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }

        return datasets, vocab_x, vocab_y

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            hlog.log("No lexicon provided using identity")
            proj = np.zeros((len(self.vocab_x), len(self.vocab_y)), dtype=np.float32)
            x_keys = list(self.vocab_x.contents.keys())
            y_keys = list(self.vocab_y.contents.keys())
            for x, x_key in enumerate(x_keys):
                if x_key in y_keys:
                    y = y_keys.index(x_key)
                    proj[x, y] = 1.0

            return np.argmax(proj, axis=1)


class ColorDataset(Seq2SeqDataset):
    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
        primitives=None,
        study=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.primitives = primitives
        self.lexicon = lexicon
        self.substitute = substitute
        self.swapables = swapables
        self.study = study
        self.references = None
        self.augmentation_probability = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

    def copy_translation(self, use_lexicon: bool = False):
        if self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            proj = np.identity(len(self.vocab_x))
            vocab_keys = list(self.vocab_x.contents.keys())
            for x, y in self.primitives:
                idx = vocab_keys.index(x[0])
                idy = vocab_keys.index(y[0])
                print("x: ", x[0], " y: ", y[0])
                proj[idx, idx] = 0
                proj[idx, idy] = 1
            return np.argmax(proj, axis=1)

    @staticmethod
    def load_dataset(
        root: str = "COLOR",
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
        full_data=True,
    ):
        input_symbols_list = set(
            [
                "dax",
                "lug",
                "wif",
                "zup",
                "fep",
                "blicket",
                "kiki",
                "tufa",
                "gazzer",
            ]
        )
        output_symbols_list = set(["RED", "YELLOW", "GREEN", "BLUE", "PURPLE", "PINK"])
        study, test = get_fig2_exp(input_symbols_list, output_symbols_list)

        if full_data:
            for sym in input_symbols_list:
                vocab_x.add(sym)
            for sym in output_symbols_list:
                vocab_y.add(sym)
            max_len_x = 7
            max_len_y = 9
        else:
            test, study = study[3:4], study[0:3]
            for x, y in test + study:
                for sym in x:
                    vocab_x.add(sym)
                for sym in y:
                    vocab_y.add(sym)
            max_len_x = 2
            max_len_y = 2

        train_items, test_items = encode(study, vocab_x, vocab_y), encode(
            test, vocab_x, vocab_y
        )
        val_items = test_items

        hlog.value("vocab_x\n", vocab_x)
        hlog.value("vocab_y\n", vocab_y)
        hlog.value("study\n", study)
        hlog.value("test\n", test)

        datasets = {
            "train": COGSDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
                primitives=study,
                study=study,
            ),
            "val": COGSDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
                primitives=study,
                study=study,
            ),
            "test": COGSDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
                primitives=study,
                study=study,
            ),
        }
        return datasets, vocab_x, vocab_y


class CoGnitionDataset(Seq2SeqDataset):
    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.substitute = substitute
        self.swapables = swapables
        self.augmentation_probability = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            hlog.log("No lexicon provided, using identity")
            proj = np.zeros((len(self.vocab_x), len(self.vocab_y)), dtype=np.float32)
            x_keys = list(self.vocab_x.contents.keys())
            y_keys = list(self.vocab_y.contents.keys())
            for x, x_key in enumerate(x_keys):
                if x_key in y_keys:
                    y = y_keys.index(x_key)
                    proj[x, y] = 1.0
            return np.argmax(proj, axis=1)

    @staticmethod
    def load_dataset(
        root: str,
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        data = {}
        max_len_x, max_len_y = 0, 0
        count_x, count_y = Counter(), Counter()

        for split in ("train", "valid", "test"):
            split_data = []
            if split == "test":
                translate_file = f"{root}/CoGnition/data/processed/cg-test.fast"
            else:
                translate_file = f"{root}/CoGnition/data/processed/{split}.fast"

            with open(translate_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    str_inp, str_out = line.replace("@@ ", "").split(
                        " ||| "
                    )  ## change this if you want to keep as bpe
                    str_inp = str_inp.strip()
                    str_out = str_out.strip()
                    inp, out = (
                        vocab_x.tokenize(str_inp),
                        vocab_y.tokenize(str_out),
                    )
                    max_len_x = max(len(inp), max_len_x)
                    max_len_y = max(len(out), max_len_y)
                    for t in inp:
                        count_x[t] += 1
                    for t in out:
                        count_y[t] += 1
                    split_data.append((inp, out))

            data[split] = split_data

        print("total vocab_x: ", len(count_x))
        print("total vocab_y: ", len(count_y))

        count_x = count_x.most_common(15000)
        count_y = count_y.most_common(31000)

        for x, _ in count_x:
            vocab_x.add(x)
        for y, _ in count_y:
            vocab_y.add(y)

        edata = {}
        references = {}
        for split, split_data in data.items():
            esplit = []
            for inp, out in split_data:
                (einp, eout) = encode_io((inp, out), vocab_x, vocab_y)
                esplit.append((einp, eout))
                sinp = " ".join(vocab_x.decode(einp))
                sinp = sinp.replace("@@ ", "")
                out = " ".join(out).replace("@@ ", "").split(" ")
                if sinp in references:
                    references[sinp].append(out)
                else:
                    references[sinp] = [out]
            edata[split] = esplit
        data = edata

        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("max length x: ", max_len_x)
        hlog.value("max length y: ", max_len_y)

        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])

        train_items = data["train"]
        val_items = data["valid"]
        test_items = data["test"]

        datasets = {
            "train": CoGnitionDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": CoGnitionDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": CoGnitionDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }

        return datasets


class CLEVRDataset(Seq2SeqDataset):
    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
        references=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.swapables = swapables
        self.substitute = substitute
        self.augmentation_probability = 0
        self.references = references

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def augmenter(self, datum):
        inputs, targets = datum["inputs"], datum["targets"]
        inputs, targets = augmentation.make_a_swap_single_t5(
            inputs,
            targets,
            self.vocab_x,
            self.vocab_y,
            self.lexicon,
            self.swapables,
            substitute=self.substitute,
        )
        return {"inputs": inputs, "targets": targets}

    def preprocess(self, datum):
        return datum["inputs"], datum["targets"]

    @staticmethod
    def load_dataset(
        root: str,
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        data = {}
        max_len_x, max_len_y = 0, 0

        if vocab_x is None:
            vocab_x = Vocab()

        if vocab_y is None:
            vocab_y = Vocab()

        for split in (
            "train",
            "val",
        ):
            split_data = []
            path = f"{root}/{split}_encodings.tsv"
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    str_text, str_answer, str_encodings, *_ = line.split("\t")
                    str_inp = "Image: " + str_encodings + " Q: " + str_text + " A:"
                    inp = vocab_x.tokenize(str_inp)
                    out = [str_answer]
                    max_len_x = max(len(inp), max_len_x)
                    max_len_y = max(len(out), max_len_y)

                    if isinstance(vocab_x, Vocab):
                        for t in inp:
                            vocab_x.add(t)
                            vocab_y.add(t)
                        for t in out:
                            vocab_y.add(t)
                            vocab_x.add(t)

                    split_data.append({"inputs": str_inp, "targets": str_answer})

            data[split] = split_data

        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])
        train_items = data["train"]
        val_items = data["val"]
        test_items = [d for d in data["val"]]

        datasets = {
            "train": CLEVRDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": CLEVRDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": CLEVRDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }

        return datasets, vocab_x, vocab_y

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            hlog.log("No lexicon provided using identity")
            proj = np.zeros((len(self.vocab_x), len(self.vocab_y)), dtype=np.float32)
            x_keys = list(self.vocab_x.contents.keys())
            y_keys = list(self.vocab_y.contents.keys())
            for x, x_key in enumerate(x_keys):
                if x_key in y_keys:
                    y = y_keys.index(x_key)
                    proj[x, y] = 1.0

            return np.argmax(proj, axis=1)


class TRANSLATEDataset(Seq2SeqDataset):
    def __init__(
        self,
        data,
        max_len_x,
        max_len_y,
        vocab_x,
        vocab_y,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
    ):
        self.data = data
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.lexicon = lexicon
        self.swapables = swapables
        self.substitute = substitute
        self.augmentation_probability = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

    def copy_translation(self, use_lexicon: bool = False):
        if use_lexicon and self.lexicon:
            return tranlate_with_lexiconv2(self.lexicon, self.vocab_x, self.vocab_y)
        else:
            hlog.log("No lexicon provided, using identity")
            proj = np.zeros((len(self.vocab_x), len(self.vocab_y)), dtype=np.float32)
            x_keys = list(self.vocab_x.contents.keys())
            y_keys = list(self.vocab_y.contents.keys())
            for x, x_key in enumerate(x_keys):
                if x_key in y_keys:
                    y = y_keys.index(x_key)
                    proj[x, y] = 1.0
            return np.argmax(proj, axis=1)

    @staticmethod
    def load_dataset(
        root: str,
        vocab_x=None,
        vocab_y=None,
        substitute: bool = False,
        lexicon=None,
        swapables=None,
        lessdata=False,
        geca=False,
        seed=0,
    ):
        data = {}
        max_len_x, max_len_y = 0, 0
        count_x, count_y = Counter(), Counter()

        for split in ("train", "dev", "test"):
            split_data = []
            if split == "train" and geca:
                translate_file = f"{root}/cmn.txt_{split}_tokenized_lexgen"
            else:
                translate_file = f"{root}/cmn.txt_{split}_tokenized"

            if lessdata and split == "train":
                translate_file = (
                    translate_file.replace("TRANSLATE", "TRANSLATE/less")
                    + f"_{seed}.tsv"
                )
            else:
                translate_file = translate_file + ".tsv"

            with open(translate_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    str_inp, str_out = line.split("\t")
                    str_inp = str_inp.strip()
                    str_out = str_out.strip()
                    inp, out = (
                        vocab_x.tokenize(str_inp),
                        vocab_y.tokenize(str_out),
                    )
                    max_len_x = max(len(inp), max_len_x)
                    max_len_y = max(len(out), max_len_y)
                    for t in inp:
                        count_x[t] += 1
                    for t in out:
                        count_y[t] += 1
                    split_data.append((str_inp, str_out))

            data[split] = split_data

        count_x = count_x.most_common(15000)  # threshold to 10k words
        count_y = count_y.most_common(26000)  # threshold to 10k words

        for x, _ in count_x:
            vocab_x.add(x)
        for y, _ in count_y:
            vocab_y.add(y)

        edata = {}
        references = {}
        for split, split_data in data.items():
            esplit = []
            for inp, out in split_data:
                (einp, eout) = encode_io((inp, out), vocab_x, vocab_y)
                esplit.append((einp, eout))
                sinp = " ".join(vocab_x.decode(einp))
                if sinp in references:
                    references[sinp].append(out)
                else:
                    references[sinp] = [out]
            edata[split] = esplit
        data = edata

        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])

        train_items = data["train"]
        val_items = data["dev"]
        test_items = data["test"]

        datasets = {
            "train": TRANSLATEDataset(
                train_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "val": TRANSLATEDataset(
                val_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
            "test": TRANSLATEDataset(
                test_items,
                max_len_x,
                max_len_y,
                vocab_x,
                vocab_y,
                substitute=substitute,
                lexicon=lexicon,
                swapables=swapables,
            ),
        }
        return datasets, vocab_x, vocab_y


DATASET_FACTORY = {
    "color": ColorDataset,
    "cognition": CoGnitionDataset,
    "clevr": CLEVRDataset,
    "translate": TRANSLATEDataset,
    "cogs": COGSDataset,
    "scan": SCANDataset,
    "alchemy": TextAlchemyDataset,
    "speech": SPEECHDataset,
}


def tranlate_with_lexiconv2(
    lexicon,
    vocab_x,
    vocab_y,
    unwanted=lambda x: False,
    soft_align=False,
    learn_align=False,
    temp=0.02,
    soft_temp=0.02,
):
    if lexicon == "uniform":
        proj = np.ones((len(vocab_x), len(vocab_y)), dtype=np.float32)
    elif lexicon == "random":
        proj = np.random.default_rng().random(
            (len(vocab_x), len(vocab_y)), dtype=np.float32
        )
    else:
        proj = np.zeros((len(vocab_x), len(vocab_y)), dtype=np.float32)

        x_keys = list(vocab_x.contents.keys())
        y_keys = list(vocab_y.contents.keys())

        for x, x_key in enumerate(x_keys):
            if x_key in y_keys:
                y = vocab_y[x_key]
                proj[x, y] = 1.0

        for w, a in lexicon.items():
            if w in vocab_x and len(a) > 0 and not unwanted(w):
                x = vocab_x[w]
                for v, n in a.items():
                    if not unwanted(v) and v in vocab_y:
                        y = vocab_y[v]
                        proj[x, y] = 2 * n

        empty_xs = np.where(proj.sum(axis=1) == 0)[0]
        empty_ys = np.where(proj.sum(axis=0) == 0)[0]

        if len(empty_ys) != 0 and len(empty_xs) != 0:
            for i in empty_xs:
                proj[i, empty_ys] = 1 / len(empty_ys)

    if soft_align:
        return SoftAlign(proj / soft_temp, requires_grad=learn_align)
    else:
        return np.argmax(proj, axis=1)


def tranlate_with_lexicon(
    lexicon,
    vocab_x,
    vocab_y,
    unwanted=lambda x: False,
    soft_align=False,
    learn_align=False,
    temp=0.02,
):
    proj = np.identity(len(vocab_x), dtype=np.float32)
    vocab_keys = list(vocab_x.contents.keys())

    for w, a in lexicon.items():
        if w in vocab_x and len(a) > 0 and not unwanted(w):
            x = vocab_keys.index(w)
            for v, n in a.items():
                if not unwanted(v) and v in vocab_y:
                    y = vocab_keys.index(v)
                    proj[x, y] = 2 * n

    if soft_align:
        return SoftAlign(proj / temp, requires_grad=learn_align)
    else:
        return np.argmax(proj, axis=1)
