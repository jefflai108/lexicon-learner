import os

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("source_file", default=None, help="source language file")
flags.DEFINE_string("target_file", default=None, help="target language file")
flags.DEFINE_string(
    "output_file",
    default=None,
    help="output language file to write merged file",
)


def add_prefix(tokens, prefix="S"):
    return [prefix + token for token in tokens]


def main(_):
    with open(FLAGS.source_file) as handle:
        source_data = [
            add_prefix(line.strip().split(" "), prefix="") for line in handle
        ]

    with open(FLAGS.target_file) as handle:
        target_data = [
            add_prefix(line.strip().split(" "), prefix="") for line in handle
        ]

    assert len(target_data) == len(source_data)

    with open(FLAGS.output_file, "w") as handle:
        for src, tgt in zip(source_data, target_data):
            print(" ".join(src) + "\t" + " ".join(tgt), file=handle)


if __name__ == "__main__":
    app.run(main)
