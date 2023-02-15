import numpy as np
import torch
from tqdm import tqdm


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecordLoss:
    def __init__(self):
        self.n_iter = 0
        self.losses = AverageMeter()
        self.nlls = AverageMeter()
        self.point_kls = AverageMeter()
        self.logprob_pyxs = AverageMeter()
        self.entropys = AverageMeter()
        self.qx_samples = []
        self.px_samples = []
        self.py_samples = []
        self.measures = (
            self.losses,
            self.nlls,
            self.point_kls,
            self.logprob_pyxs,
            self.entropys,
        )

    def reset(self):
        self.__init__()

    def add_samples_qx(self, samples):
        self.qx_samples += [samples]

    def add_samples_py(self, samples):
        self.py_samples += [samples]

    def add_samples_px(self, samples):
        self.px_samples += [samples]

    def reset_samples(self):
        self.qx_samples = []
        self.px_samples = []
        self.py_samples = []

    def update(
        self, loss=None, nll=None, point_kl=None, logprob_pyx=None, entropy=None
    ):
        for i, term in enumerate((loss, nll, point_kl, logprob_pyx, entropy)):
            if term is not None:
                self.measures[i].update(term[0], term[1])
        self.n_iter += 1


def batch_seqs(seqs):
    max_len = max(len(s) for s in seqs)
    data = np.zeros((max_len, len(seqs)))
    for i, s in enumerate(seqs):
        data[: len(s), i] = s
    return torch.LongTensor(data)


def weight_top_p(vec, p):
    indices = (-vec).argsort()
    out = np.zeros_like(vec)
    cumprob = 0
    for i in indices:
        excess = max(0, cumprob + vec[i] - p)
        weight = vec[i] - excess
        out[i] = weight
        cumprob += weight
        if excess > 0:
            break

    out /= out.sum()
    return out


def trim(L, obj):
    if obj in L:
        return L[: L.index(obj) + 1]
    return L


"""Noam Scheduler."""


from torch.optim import lr_scheduler


class NoamLR(lr_scheduler._LRScheduler):
    r"""Noam Learning rate schedule.
    Increases the learning rate linearly for the first `warmup_steps` training steps, then decreases it proportional to
    the inverse square root of the step number.
              ^
             / \
            /   `
           /     `
          /         `
         /               `
        /                       `
       /                                   `
      /                                                    `
     /                                                                              `
    /                                                                                                                  `
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.
    Notes
    -----
    If step <= warmup_steps,
        scale = step / warmup_steps
    If step > warmup_steps,
        scale = (warmup_steps ^ 0.5) / (step ^ 0.5)
    """

    def __init__(self, optimizer, model_size, warmup_steps=4000):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer)

    def scale(self, step):
        return self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )

    def get_lr(self):
        scale = self.scale(max(1, self._step_count))
        return [base_lr * scale for base_lr in self.base_lrs]


PRETRAINED_EMBEDDING_CACHE = None


def clean_embedding_cache():
    global PRETRAINED_EMBEDDING_CACHE
    PRETRAINED_EMBEDDING_CACHE = None


def load_from_saved_embeddings(file, vocab, embeddings, allwords):
    global PRETRAINED_EMBEDDING_CACHE
    if PRETRAINED_EMBEDDING_CACHE is None:
        cache = {}
        print(f"Loading pretrained embeddings from {file}")
        with open(file) as f:
            for l in tqdm(f):
                word, *vector = l.strip().split()
                if word in allwords:
                    cache[word] = np.array(list(map(np.float32, vector)))

        PRETRAINED_EMBEDDING_CACHE = cache
    else:
        print("Loading pretrained embeddings from cache")

    word2vec = np.array(list(PRETRAINED_EMBEDDING_CACHE.values()))
    mean = np.mean(word2vec, axis=0)
    cov = np.cov(word2vec.transpose())

    for word in vocab.contents.keys():
        id = vocab[word]
        vector = PRETRAINED_EMBEDDING_CACHE.get(word.lower(), None)
        with torch.no_grad():
            if vector is None:
                embeddings[id, :] = torch.tensor(
                    np.random.multivariate_normal(mean, cov, size=1),
                    device=embeddings.device,
                )
            else:
                embeddings[id, :] = torch.tensor(vector, device=embeddings.device)

    return embeddings
