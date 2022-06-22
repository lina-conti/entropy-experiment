from typing import List

import argparse

import pandas as pd

import torch
from torch import Tensor, IntTensor
from torch.nn import LogSoftmax, Softmax

from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import load_config, get_latest_checkpoint, load_checkpoint
from joeynmt.model import build_model, Model
from joeynmt.constants import EOS_TOKEN, BOS_TOKEN, UNK_TOKEN

from halo import Halo
import tqdm

import json
import datetime

from math import exp, log
import numpy as np
import seaborn as sns

from scipy.stats import entropy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

log_softmax = LogSoftmax(dim=1)
softmax = Softmax(dim=1)


def load_model(cfg_file: str):

    with Halo(text="Loading configuration", spinner="dots") as spinner:
        cfg = load_config(cfg_file)
    spinner.succeed("Loading configuration")

    model_dir = cfg["training"]["model_dir"]

    # read vocabs
    with Halo(text="Loading vocabulary", spinner="dots") as spinner:
        src_vocab_file = cfg["data"].get("src_vocab",
                                         model_dir + "/src_vocab.txt")
        trg_vocab_file = cfg["data"].get("trg_vocab",
                                         model_dir + "/trg_vocab.txt")
        src_vocab = Vocabulary(file=src_vocab_file)
        trg_vocab = Vocabulary(file=trg_vocab_file)
    spinner.succeed("Loading vocabulary")

    with Halo(text='Loading model', spinner='dots') as spinner:
        ckpt = get_latest_checkpoint(model_dir)
        model_checkpoint = load_checkpoint(ckpt, use_cuda=False)
        model = build_model(cfg["model"],
                            src_vocab=src_vocab,
                            trg_vocab=trg_vocab)
        model.load_state_dict(model_checkpoint["model_state"])
    spinner.succeed("Loading model")

    return model


def decoding(
        model: Model,
        encoder_output: Tensor):

    """
    Parameters
    ----------
    - model: a JoeyNMT model
    - encoder_output: the encoded sentence built by the encoder

    Returns
    -------
    A list of ids of predicted tokens
    AND
    A DataFrame that has as many rows as tokens in the reference
    sentence. The i-th row describe the prediction of the i-th token:
    - the id of the predicted token in the vocabulary and the
      probability of generating it (given the source sentence and the
      predicted prefix)
    - the log probability distribution (a list the i-th entry of which
      correspond to the probability of generating the i-th token of
      the vocabulary)
    - the probability distribution
    - the entropy of the probability distribution
    """
    predicted_translation = []

    # As we consider a single sentence, there is no need for padding
    # and we can always consider all the words in the input
    # sentence
    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

    bos_index = model.bos_index
    eos_index = model.eos_index

    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

    res = []
    # loop on the max size of sentence
    while(True):
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
            probas = softmax(logits)

            max_value, pred_trg_token = torch.max(logits, dim=1)
            pred_trg_token = pred_trg_token.data.unsqueeze(-1)
            res.append({"predicted_token_idx": pred_trg_token.item(),
                        "predicted_log_proba": log_probas[0][pred_trg_token].item(),
                        "log_probas": log_probas[0].detach().cpu().numpy(),
                        "probas": probas[0].detach().cpu().numpy(),
                        "entropy": entropy(probas[0].detach().cpu().numpy())
                        })

            ys = torch.cat([ys, IntTensor([[pred_trg_token]])], dim=1)
            # print(ys)
            if(pred_trg_token == eos_index):
                break

            predicted_translation.append(pred_trg_token.item())

    return predicted_translation, pd.DataFrame(res)


def encode_sentence(sentence: List[str], model):

    indexes = [model.src_vocab.stoi[token] for token in sentence + [EOS_TOKEN]]
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

def to_tokens(predicted_ids, model):
    sentence = ""
    for id in predicted_ids:
        sentence += model.trg_vocab.itos[id] + " "
    return sentence

def difference_highest_second(prob_distribution: np.ndarray):
    parted_list = np.partition(prob_distribution, -2)
    return parted_list[-1] - parted_list[-2]
