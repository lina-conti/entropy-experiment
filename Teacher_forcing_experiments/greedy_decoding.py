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
import numpy as np

from pathlib import Path
from joeynmt.tokenizers import build_tokenizer

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
        encoder_output: Tensor,
        max_output_size: int):
    """
    Parameters
    ----------
    - model: a JoeyNMT model
    - encoder_output: the encoded sentence built by the encoder
    - max_output_length: max sentence length (to prevent infinite loops)

    Returns
    The list of predicted ids
    """
    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

    bos_index = model.bos_index
    eos_index = model.eos_index

    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

    res = []
    for i in range(max_output_length):
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

            ys = torch.cat([ys, IntTensor([[pred_trg_token]])], dim=1)

            if pred_trg_token == eos_index:
                break
            res.append(int(pred_trg_token))

    return res


def encode_sentence(sentence: List[str], model):

    indexes = [model.bos_index] + [model.src_vocab.lookup(token) for token in sentence + [EOS_TOKEN]]
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
    cfg = load_config("/home/lina/Desktop/Stage/transformer_2.0/wmt15_fra2eng.yaml")
    max_output_length = cfg["testing"]["max_output_length"]
    tokenizer = build_tokenizer(cfg["data"])

    #s = "▁l ' athlète ▁a ▁terminé ▁son ▁travail ▁."
    #t = "▁the ▁athlete ▁finished ▁his ▁work ▁."
    s = "l'athlète a terminé son travail."
    src_tokenizer = tokenizer[cfg["data"]["src"]["lang"]]
    s_tokenized = src_tokenizer(src_tokenizer.pre_process(s))

    src = encode_sentence(s_tokenized, model)

    res = greedy_decoding(model, src, max_output_length)

    print(model.trg_vocab.array_to_sentence(np.array(res)))
