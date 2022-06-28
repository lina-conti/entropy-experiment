#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#                    Version 2, December 2004

# Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

# Everyone is permitted to copy and distribute verbatim or modified
# copies of this license document, and changing it is allowed as long
# as the name is changed.

#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

#  0. You just DO WHAT THE FUCK YOU WANT

"""
Simple functions to realize a forced decoding with a JoeyNMT model

There are three functions:
- `load_model` that loads a JoeyNMT model;
- `encode_sentence` that take a tokenized sentence and return the
   representation built for each of its tokens;
- `forced_decoding`
"""


from typing import List

import pandas as pd

import torch
from torch import Tensor, IntTensor
from torch.nn import LogSoftmax

from joeynmt.vocabulary import build_vocab
from joeynmt.helpers import load_config, get_latest_checkpoint, load_checkpoint
from joeynmt.model import build_model, Model
from joeynmt.constants import EOS_TOKEN, BOS_TOKEN, UNK_TOKEN

from halo import Halo
import tqdm

from pathlib import Path

log_softmax = LogSoftmax(dim=1)


def load_model(cfg_file: str):

    with Halo(text="Loading configuration", spinner="dots") as spinner:
        cfg = load_config(cfg_file)
    spinner.succeed("Loading configuration")

    model_dir = cfg["training"]["model_dir"]

    # read vocabs
    with Halo(text="Loading vocabulary", spinner="dots") as spinner:
        src_vocab, trg_vocab = build_vocab(cfg["data"])
    spinner.succeed("Loading vocabulary")

    with Halo(text='Loading model', spinner='dots') as spinner:
        ckpt = get_latest_checkpoint(Path(model_dir))
        model_checkpoint = load_checkpoint(ckpt, torch.device('cpu'))
        model = build_model(cfg["model"],
                            src_vocab=src_vocab,
                            trg_vocab=trg_vocab)
        model.load_state_dict(model_checkpoint["model_state"])
    spinner.succeed("Loading model")

    return model


def greedy_decoding(
        model: Model,
        encoder_output: Tensor):
    """
    Parameters
    ----------
    - model: a JoeyNMT model
    - encoder_output: the encoded sentence built by the encoder
    - target: the tokenized target sentence

    Returns
    -------
    A DataFrame that has as many rows as tokens in the reference
    sentence. The i-th row describe the prediction of the i-th token:
    - the id of the reference token in the vocabulary and the
      probability of generating it (given the source sentence and the
      gold prefix)
    - the id of the predicted token in the vocabulary and the
      probability of generating it (given the source sentence and the
      gold prefix)
    - the probability distribution (a list the i-th entry of which
      correspond to the probability of generating the i-th token of
      the vocabulary)
    """
    # As we consider a single sentence, there is no need for padding
    # and we can always consider all the words in the input
    # sentence
    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

    target = [model.trg_vocab.lookup(token) for token in target + [EOS_TOKEN]]

    bos_index = model.bos_index
    eos_index = model.eos_index

    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

    res = []
    for gold_trg_token in target:
        model.eval()
        with torch.no_grad():
            logits, _, _, _ = model(
                return_type="decode",
                trg_input=ys,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                decoder_hidden=None,
                trg_mask=trg_mask
            )

            logits = logits[:, -1]
            log_probas = log_softmax(logits)

            max_value, pred_trg_token = torch.max(logits, dim=1)
            pred_trg_token = pred_trg_token.data.unsqueeze(-1)
            res.append({"predicted_token_idx": pred_trg_token.item(),
                        "predicted_log_proba": log_probas[0][pred_trg_token].item(),
                        "gold_token_idx": gold_trg_token,
                        "gold_log_proba": log_probas[0][gold_trg_token].item(),
                        "log_probas": log_probas[0].detach().cpu().numpy()
                        })

            ys = torch.cat([ys, IntTensor([[pred_trg_token]])], dim=1)

    return pd.DataFrame(res)


def encode_sentence(sentence: List[str], model):

    indexes = [model.src_vocab.lookup(token) for token in sentence + [EOS_TOKEN]]
    # list of lists because input to the NN has to be a list of sentences
    src = torch.tensor([indexes])
    lengths = torch.tensor([len(indexes)])
    masks = torch.tensor([[[True for _ in range(len(indexes))]]])

    model.eval()
    with torch.no_grad():
        encoder_output, _, _, _ = model(return_type="encode",
                                        src=src,
                                        src_length=lengths,
                                        src_mask=masks)

    return encoder_output


if __name__ == "__main__":
    model = load_model("/home/lina/Desktop/Stage/transformer_2.0/wmt15_fra2eng.yaml")

    s = "▁l ' athlète ▁a ▁terminé ▁son ▁travail ▁."
    t = "▁the ▁athlete ▁finished ▁his ▁work ▁."

    src = encode_sentence(s.split(), model)

    df = greedy_decoding(model, src)

    print(model.trg_vocab.array_to_sentence(df['predicted_token_idx'].to_numpy()))
