from typing import List, Dict
import torch
from torch import Tensor, IntTensor
from torch.nn import LogSoftmax, Softmax
log_softmax = LogSoftmax(dim=1)

from joeynmt.vocabulary import build_vocab
from joeynmt.helpers import load_config, get_latest_checkpoint, load_checkpoint
from joeynmt.tokenizers import build_tokenizer
from joeynmt.model import build_model, Model
from joeynmt.constants import EOS_TOKEN, BOS_TOKEN, UNK_TOKEN

from halo import Halo
import argparse
from pathlib import Path
import datetime
import numpy as np
import itertools
import pandas as pd
from scipy.stats import entropy

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

def predict_token(encoder_output: Tensor, history: Tensor, src_mask: Tensor,
                                        trg_mask:Tensor, model: Model, return_log_probs=False):
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(
            return_type="decode",
            trg_input=history,
            encoder_output=encoder_output,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=None,
            decoder_hidden=None,
            trg_mask=trg_mask
        )
    logits = logits[:, -1]
    max_value, pred_trg_token = torch.max(logits, dim=1)
    pred_trg_token = pred_trg_token.data.unsqueeze(-1)
    if return_log_probs:
        return int(pred_trg_token), log_softmax(logits)
    return int(pred_trg_token)

def predict_wrong_token(gold_trg_token: int, encoder_output: Tensor,
        history: Tensor, src_mask: Tensor, trg_mask:Tensor, model: Model) -> int:
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(
            return_type="decode",
            trg_input=history,
            encoder_output=encoder_output,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=None,
            decoder_hidden=None,
            trg_mask=trg_mask
        )
    logits = logits[:, -1]
    while(True):
        max_value, pred_trg_token = torch.max(logits, dim=1)
        pred_trg_token = int(pred_trg_token.data.unsqueeze(-1))
        if pred_trg_token != gold_trg_token:
            return pred_trg_token
        logits[0][pred_trg_token] = float('-inf')
