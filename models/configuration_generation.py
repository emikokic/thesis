# -*- coding: utf-8 -*-
# Author: Cristian Cardellino

from __future__ import absolute_import, print_function

import argparse
import json
import math

from hashlib import md5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path")
    parser.add_argument("--convs",
                        nargs="+",
                        default=None,
                        type=str)
    parser.add_argument("--denses",
                        nargs="*",
                        default=[],
                        type=int)
    parser.add_argument("--max-sentence-len",
                        default=100,
                        type=int)
    parser.add_argument("--embeddings-size",
                        default=50,
                        type=int)
    parser.add_argument("--noise-std",
                        default=0.3,
                        type=float)
    parser.add_argument("--starter-learning-rate",
                        default=0.02,
                        type=float)
    parser.add_argument("--decay-after",
                        default=5,
                        type=int)
    parser.add_argument("--vocab-size",
                        default=None,
                        type=int)
    parser.add_argument("--num-classes",
                        default=None,
                        type=int)
    parser.add_argument("--epochs",
                        default=10,
                        type=int)
    parser.add_argument("--batch-size",
                        default=100,
                        type=int)

    args = parser.parse_args()

    convs = args.convs
    assert convs is not None

    denses = args.denses
    max_sentence_len = args.max_sentence_len
    embeddings_size = args.embeddings_size
    noise_std = args.noise_std
    starter_learning_rate = args.starter_learning_rate
    decay_after = args.decay_after
    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = args.num_classes
    assert num_classes is not None
    vocab_size = args.vocab_size
    assert vocab_size is not None

    layers = []
    current_filter_size = embeddings_size
    current_sequence_size = max_sentence_len
    for lidx, conv in enumerate(convs):
        spec = conv.split(":")

        assert lidx > 0 or spec[0] == "conv", "The first layer must be a convolution"
        assert spec[0] in {"conv", "max_pool"}

        layer_spec = {
            "name": "%s%d" % (spec[0], lidx),
            "type": spec[0]
        }

        if spec[0] == "conv":
            layer_spec["kernel_size"] = int(spec[1])
            layer_spec["input_filters"] = current_filter_size
            layer_spec["output_filters"] = int(spec[2])
            current_filter_size = layer_spec["output_filters"]
        elif spec[0] == "max_pool":
            layer_spec["pool_size"] = int(spec[1])
            assert current_sequence_size % layer_spec["pool_size"] == 0
            current_sequence_size = current_sequence_size // layer_spec["pool_size"]

        layers.append(layer_spec)

    layers.append({"name": "flat", "type": "flat"})
    current_dense_shape = current_sequence_size * current_filter_size

    for lidx, dense_shape in enumerate(denses, start=len(convs)):
        layers.append({
            "name": "dense%d" % lidx,
            "type": "dense",
            "shape": [current_dense_shape, dense_shape]
        })
        current_dense_shape = dense_shape

    layers.append({
        "name": "predictions",
        "type": "dense",
        "shape": [current_dense_shape, num_classes]
    })

    denoising_cost = [0.01 for _ in range(len(layers) + 1)]
    denoising_cost[0] = 1000
    denoising_cost[1] = 10

    configuration = dict(
        hyperparameters=",".join([
            "-".join(convs),
            "-".join([str(d) for d in denses]),
            str(max_sentence_len),
            str(embeddings_size),
            str(vocab_size),
            str(noise_std),
            str(starter_learning_rate),
            str(decay_after),
            str(num_classes),
            str(batch_size),
            str(epochs)
        ]),
        layers=layers,
        max_sentence_len=max_sentence_len,
        embeddings_size=embeddings_size,
        vocab_size=vocab_size,
        denoising_cost=denoising_cost,
        noise_std=noise_std,
        starter_learning_rate=starter_learning_rate,
        decay_after=decay_after,
        num_classes=num_classes,
        num_labeled=batch_size,
        batch_size=batch_size,
        num_epochs=epochs
    )
    configuration["experiment_id"] = md5(configuration["hyperparameters"].encode("utf-8")).hexdigest()

    with open("%s/%s.json" % (args.save_path, configuration["experiment_id"]), "w") as fh:
        json.dump(configuration, fh)
