import json
import os
import random

from absl import app
from absl import flags
import hlog
import matplotlib.pyplot as plt
from mutex import Mutex
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import seaborn as sns
from src import dataset_utils
from src import vocab
from src.multiiter import MultiIter
from src.projection import SoftAlign
from src.utils import NoamLR
from src.utils import RecordLoss
import torch
from torch import optim
import torch.utils.data as torch_data
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb


sns.set()

FLAGS = flags.FLAGS
flags.DEFINE_integer("dim", default=512, help="trasnformer dimension")
flags.DEFINE_integer("n_layers", default=2, help="number of rnn layers")
flags.DEFINE_integer("n_batch", default=512, help="batch size")
flags.DEFINE_float("gclip", default=0.5, help="gradient clip")
flags.DEFINE_integer("n_epochs", default=100, help="number of training epochs")
flags.DEFINE_integer("beam_size", default=5, help="beam search size")
flags.DEFINE_float("lr", default=1.0, help="learning rate")
flags.DEFINE_float("temp", default=1.0, help="temperature for samplings")
flags.DEFINE_float("dropout", default=0.4, help="dropout")
flags.DEFINE_string("save_model", default="model.m", help="model save location")
flags.DEFINE_string("load_model", default="", help="load pretrained model")
flags.DEFINE_integer("seed", default=0, help="random seed")
flags.DEFINE_bool("debug", default=False, help="debug mode")
flags.DEFINE_bool(
    "full_data",
    default=True,
    help="full figure 2 experiments or simple color dataset",
)
flags.DEFINE_string("dataset", default="alchemy", help="dataset")
flags.DEFINE_bool("bidirectional", default=False, help="bidirectional encoders")
flags.DEFINE_bool("attention", default=True, help="Source Attention")
flags.DEFINE_integer("warmup_steps", default=4000, help="noam warmup_steps")
flags.DEFINE_integer("valid_steps", default=500, help="validation steps")
flags.DEFINE_integer("max_step", default=8000, help="maximum number of steps")
flags.DEFINE_integer("tolarance", default=5, help="early stopping tolarance")
flags.DEFINE_integer("accum_count", default=4, help="grad accumulation count")
flags.DEFINE_bool("shuffle", default=True, help="shuffle training set")
flags.DEFINE_bool("lr_schedule", default=True, help="noam lr scheduler")
flags.DEFINE_string("scan_split", default="around_right", help="around_right or jump")
flags.DEFINE_bool("qxy", default=True, help="train pretrained qxy")
flags.DEFINE_bool("copy", default=False, help="enable copy mechanism")
flags.DEFINE_bool("highdrop", default=False, help="high drop mechanism")
flags.DEFINE_bool("highdroptest", default=False, help="high drop at test")
flags.DEFINE_float("highdropvalue", default=0.5, help="high drop value")
flags.DEFINE_string("aligner", default="", help="alignment file by fastalign")
flags.DEFINE_bool("soft_align", default=False, help="lexicon projection matrix")
flags.DEFINE_bool("geca", default=False, help="use geca files for translate")
flags.DEFINE_bool("lessdata", default=False, help="0.1 data for translate")
flags.DEFINE_bool(
    "learn_align", default=False, help="learned lexicon projection matrix"
)
flags.DEFINE_float("p_augmentation", default=0.0, help="augmentation ratio")
flags.DEFINE_string("aug_file", default="", help="data source for augmentation")
flags.DEFINE_float("soft_temp", default=0.2, help="2 * temperature for soft lexicon")
flags.DEFINE_string("tb_dir", default="", help="tb_dir")
flags.DEFINE_integer("gpu", default=0, help="gpu id")
flags.DEFINE_string("swapables", default=None, help="swapables json")
flags.DEFINE_string("embedding_file", default=None, help="word2vec file")
flags.DEFINE_bool("substitute", default=False, help="substitute")
flags.DEFINE_bool("T5", default=False, help="T5 model")
flags.DEFINE_string("exp_name", default="Aligner", help="Experiment name")

plt.rcParams["figure.dpi"] = 300

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))


def train(
    model,
    train_dataset,
    val_dataset,
    p_augmentation: float = 0.0,
):
    hlog.log(f"augmentation: {p_augmentation}")
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.998))

    if FLAGS.lr_schedule:
        scheduler = NoamLR(opt, FLAGS.dim, warmup_steps=FLAGS.warmup_steps)
    else:
        scheduler = None

    if p_augmentation > 0:
        if train_dataset.swapables:
            hlog.log(f"Enabling augmentation with ratio {p_augmentation}")
            train_dataset.enable_augmentation(p_augmentation)
        else:
            hlog.log("no lexicon + type provided for swap augmentation")

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=FLAGS.n_batch,
        shuffle=FLAGS.shuffle,
        collate_fn=train_dataset.collate_fn,
    )

    tolarance = FLAGS.tolarance
    best_f1 = best_acc = -np.inf
    best_loss = np.inf
    best_bleu = steps = accum_steps = 0
    got_nan = False
    is_running = lambda: not got_nan and accum_steps < FLAGS.max_step and tolarance > 0

    while is_running():
        train_loss = train_batches = 0
        opt.zero_grad()
        recorder = RecordLoss()

        for datum in train_loader:
            if not is_running():
                break

            inputs = datum["inputs"]
            targets = datum["targets"]

            if isinstance(model.vocab_x, vocab.Vocab):
                nll = model(
                    inputs.to(DEVICE),
                    targets.to(DEVICE),
                    lens=datum["lengths"],
                    recorder=recorder,
                )
            else:
                targets[targets == model.vocab_y.pad_token_id] = -100

                nll = model.pyx(
                    inputs.to(DEVICE),
                    targets.to(DEVICE),
                    inp_mask=datum["inputs_mask"].to(DEVICE),
                    out_mask=datum["targets_mask"].to(DEVICE),
                )

            steps += 1
            loss = nll / FLAGS.accum_count
            loss.backward()
            train_loss += loss.detach().item() * FLAGS.accum_count
            train_batches += 1

            if steps % FLAGS.accum_count == 0:
                accum_steps += 1
                opt.step()
                opt.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if accum_steps % FLAGS.valid_steps == 0:
                    with hlog.task(accum_steps):
                        hlog.value("curr loss", train_loss / train_batches)
                        acc, f1, val_loss, bscore = validate(
                            model,
                            val_dataset,
                            references=train_dataset.references,
                            eval_name="validation",
                        )

                        model.train()
                        hlog.value("acc", acc)
                        hlog.value("f1", f1)
                        hlog.value("bscore", bscore)
                        best_f1 = max(best_f1, f1)
                        best_acc = max(best_acc, acc)
                        best_bleu = max(best_bleu, bscore)
                        hlog.value("val_loss", val_loss)
                        hlog.value("best_acc", best_acc)
                        hlog.value("best_f1", best_f1)
                        hlog.value("best_bleu", best_bleu)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            tolarance = FLAGS.tolarance
                        else:
                            tolarance -= 1
                        hlog.value("best_loss", best_loss)

    hlog.value("final_acc", acc)
    hlog.value("final_f1", f1)
    hlog.value("final_bleu", bscore)
    hlog.value("best_acc", best_acc)
    hlog.value("best_f1", best_f1)
    hlog.value("best_loss", best_loss)
    hlog.value("best_bleu", best_bleu)
    return acc, f1, bscore


def validate(
    model,
    val_dataset,
    vis=False,
    final=False,
    references=None,
    eval_name="validation",
):
    model.eval()

    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=FLAGS.n_batch,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    total = correct = loss = tp = fp = fn = 0
    cur_references = []
    candidates = []
    with torch.no_grad():
        for datum in val_loader:
            if not isinstance(model.vocab_x, vocab.Vocab):
                inputs = datum["inputs"]
                targets = datum["targets"]
                inputs_mask = datum["inputs_mask"]
                targets_mask = datum["targets_mask"]

                pred = model.pyx.sample(
                    inputs.to(DEVICE),
                    model.MAXLEN_Y,
                    inp_mask=inputs_mask.to(DEVICE),
                )
                targets[targets == model.vocab_y.pad_token_id] = -100
                loss += (
                    model.pyx(
                        inputs.to(DEVICE),
                        targets.to(DEVICE),
                        inp_mask=inputs_mask.to(DEVICE),
                        out_mask=targets_mask.to(DEVICE),
                    ).item()
                    * inputs.shape[1]
                )
                targets[targets == -100] = model.vocab_y.pad_token_id
            else:
                inputs = datum["inputs"]
                targets = datum["targets"]
                lengths = datum["lengths"]
                pred, _ = model.sample(
                    inputs.to(DEVICE),
                    lens=lengths,
                    temp=1.0,
                    max_len=model.MAXLEN_Y,
                    greedy=True,
                    beam_size=FLAGS.beam_size * final,
                    calc_score=False,
                )

                loss += (
                    model.pyx(
                        inputs.to(DEVICE), targets.to(DEVICE), lens=lengths
                    ).item()
                    * inputs.shape[1]
                )

            for i, seq in enumerate(pred):
                if not isinstance(model.vocab_y, vocab.Vocab):
                    ref = targets[i].numpy().tolist()
                else:
                    ref = targets[:, i].numpy().tolist()
                ref = val_dataset.eval_format(ref, model.vocab_y)
                pred_here = val_dataset.eval_format(pred[i], model.vocab_y)

                # print("ref: ", ref)
                # print("pred_here: ", pred_here)

                if references is None:
                    cur_references.append([ref])
                else:
                    inpref = " ".join(
                        model.vocab_x.decode(inputs[0 : lengths[i], i].numpy().tolist())
                    )
                    inpref = inpref.replace("@@ ", "")
                    cur_references.append(references[inpref])

                candidates.append(" ".join(pred_here).replace("@@ ", "").split(" "))
                correct_here = pred_here == ref
                correct += correct_here
                tp_here = len([p for p in pred_here if p in ref])
                tp += tp_here
                fp_here = len([p for p in pred_here if p not in ref])
                fp += fp_here
                fn_here = len([p for p in ref if p not in pred_here])
                fn += fn_here
                total += 1
                if vis:
                    with hlog.task(total):
                        hlog.value("label", correct_here)
                        hlog.value("tp", tp_here)
                        hlog.value("fp", fp_here)
                        hlog.value("fn", fn_here)
                        if not isinstance(model.vocab_x, vocab.Vocab):
                            inp_lst = inputs[i].detach().cpu().numpy().tolist()
                        else:
                            inp_lst = inputs[:, i].detach().cpu().numpy().tolist()
                        hlog.value(
                            "input",
                            val_dataset.eval_format(inp_lst, model.vocab_x),
                        )
                        hlog.value("gold", ref)
                        hlog.value("pred", pred_here)

    bleu_score = corpus_bleu(cur_references, candidates)
    acc = correct / total
    loss = loss / total
    if tp + fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    rec = tp / (tp + fn)
    if prec == 0 or rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    hlog.value("acc", acc)
    hlog.value("f1", f1)
    hlog.value("bleu", bleu_score)

    wandb.log(
        {
            f"{eval_name}/acc": acc,
            f"{eval_name}/f1": f1,
            f"{eval_name}/bleu_score": bleu_score,
            f"{eval_name}/loss": loss,
        }
    )

    return acc, f1, loss, bleu_score


def swap_io(items):
    return [(y, x) for (x, y) in items]


def main(argv):
    hlog.flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    global DEVICE
    DEVICE = torch.device((f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu"))
    print("Using device: ", DEVICE)

    flags_dict = hlog.flags_to_args()

    wandb.init(
        # Set the project where this run will be logged
        project="align",
        # Track hyperparameters and run metadata
        name=FLAGS.exp_name,
        config=flags_dict,
    )

    tokenizer = None
    vocab_x = None
    vocab_y = None
    lexicon = None
    swapables = None
    extra_dataset_args = {}

    if FLAGS.dataset == "translate":
        extra_dataset_args["geca"] = FLAGS.geca
        extra_dataset_args["lessdata"] = FLAGS.lessdata
    elif FLAGS.dataset == "color":
        extra_dataset_args["full_data"] = FLAGS.full_data
    elif FLAGS.dataset == "scan":
        extra_dataset_args["subset"] = FLAGS.scan_split

    if FLAGS.T5:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        vocab_x = vocab_y = tokenizer

    if FLAGS.swapables:
        with open(FLAGS.swapables, "r") as handle:
            lex_and_swaps = json.load(handle)
            lexicon = lex_and_swaps["lexicon"]
            swapables = lex_and_swaps["swapables"]

    elif FLAGS.aligner:
        with open(FLAGS.aligner, "r") as handle:
            lexicon = json.load(handle)

    if FLAGS.dataset.startswith("cogs"):
        dataset, version = FLAGS.dataset.split("_")
        FLAGS.dataset = dataset
        if version == "synth":
            extra_dataset_args[
                "root"
            ] = "/raid/lingo/akyurek/git/align/cogs-with-pretraining/data/random_cvcv_str_shorter/"
            hlog.log("Using synthetic version of cogs")
        else:
            hlog.log("Using original version of cogs")

    if FLAGS.dataset.startswith("clevr"):
        dataset, version = FLAGS.dataset.split("_")
        FLAGS.dataset = dataset
        if version == "cogent":
            extra_dataset_args[
                "root"
            ] = "/afs/csail.mit.edu/u/a/akyurek/akyurek/git/align/CLEVR/cogent"
        else:
            extra_dataset_args[
                "root"
            ] = "/afs/csail.mit.edu/u/a/akyurek/akyurek/git/align/CLEVR/clevr"

    hlog.log(f"Extra dataset args:\n{extra_dataset_args}")

    datasets, vocab_x, vocab_y = dataset_utils.DATASET_FACTORY[
        FLAGS.dataset
    ].load_dataset(
        vocab_x=vocab_x,
        vocab_y=vocab_y,
        lexicon=lexicon,
        swapables=swapables,
        substitute=FLAGS.substitute,
        **extra_dataset_args,
    )

    if not FLAGS.load_model:
        model = Mutex(
            vocab_x,
            vocab_y,
            300,
            FLAGS.dim,
            max_len_x=datasets["train"].max_len_x,
            max_len_y=datasets["train"].max_len_y,
            copy=FLAGS.copy,
            n_layers=FLAGS.n_layers,
            self_att=False,
            attention=FLAGS.attention,
            dropout=FLAGS.dropout,
            temp=FLAGS.temp,
            qxy=FLAGS.qxy,
            bidirectional=FLAGS.bidirectional,  # TODO remember human data was bidirectional
            embedding_file=FLAGS.embedding_file,
        ).to(DEVICE)

        if FLAGS.copy:
            copy_projection_matrix = datasets["train"].copy_translation(
                use_lexicon=(FLAGS.aligner != "")
            )
            if isinstance(copy_projection_matrix, SoftAlign):
                copy_projection_matrix = copy_projection_matrix.to(DEVICE)
            model.pyx.decoder.copy_translation = copy_projection_matrix

        with hlog.task("train model"):
            acc, f1, bscore = train(
                model,
                datasets["train"],
                datasets["test"],
                p_augmentation=FLAGS.p_augmentation,
            )
            hlog.log(f"acc: {acc}, f1: {f1}, bscore: {bscore}")
    else:
        model = torch.load(FLAGS.load_model)

    with hlog.task("train evaluation"):
        validate(model, datasets["train"], vis=False, eval_name="train")

    with hlog.task("val evaluation"):
        validate(model, datasets["val"], vis=True, eval_name="validation")

    with hlog.task("test evaluation (greedy)"):
        validate(model, datasets["test"], vis=True, final=False, eval_name="test")

    # with hlog.task("test evaluation (beam)"):
    #     validate(model, test_items, vis=False, final=True)

    # torch.save(model, f"seed_{FLAGS.seed}_" + FLAGS.save_model)


if __name__ == "__main__":
    app.run(main)
