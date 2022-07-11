import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

def predict_token(encoder_output: Tensor, history: Tensor, src_mask: Tensor,
                                        trg_mask:Tensor, model: Model) -> int:
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

def history_one_mistake(gold_history: Tensor, predicted_history: Tensor) -> Tensor:
    differences = gold_history != predicted_history
    indices = differences.nonzero()
    if not indices.numel():
        return gold_history
    rng = np.random.default_rng()
    i = rng.choice(indices)[1]
    new_history = gold_history.detach().clone()
    new_history[0][i] = predicted_history[0][i]
    return new_history


def mistake_stats(src_corpus: str, trg_corpus: str, pred_corpus: str, model: Model,
                  max_output_length: int) -> (pd.DataFrame, int, int):

    bos_index = model.bos_index
    only_gold = 0
    only_predicted = 0
    df = pd.DataFrame([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], ["-", "-", "-", 0, 0, "-"]],
                    index=["correct predictions", "incorrect predictions", "standard deviation"],
                    columns=["gold history", "greedy history", "beam search history",
                        "1 greedy mistake", "1 beam search mistake", "last token mistake"])

    with open(src_corpus) as f_src:
        with open(trg_corpus) as f_trg:
            with open(pred_corpus) as f_pred:
                s = 0
                for sentence_fr, sentence_en, sentence_pred in zip(f_src, f_trg, f_pred):
                    s += 1
                    with Halo(text=f"translating sentence {s}", spinner="dots"):

                        encoder_output = encode_sentence(sentence_fr.strip().split(), model)

                        src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

                        gold_target = [model.trg_vocab.stoi[token] for token in sentence_en.strip().split() + [EOS_TOKEN]]

                        pred_target = [model.trg_vocab.stoi[token] for token in sentence_pred.strip().split() + [EOS_TOKEN]]

                        ys = ys_gold = ys_last = ys_1mistake = ys_bs = encoder_output.new_full(
                                                    [1, 1], bos_index, dtype=torch.long)
                        trg_mask = src_mask.new_ones([1, 1, 1])

                        one_mistake_correct = np.zeros(10)
                        one_mistake_incorrect = np.zeros(10)

                        for gold_trg_token, bs_trg_token in itertools.zip_longest(gold_target, pred_target):
                            if gold_trg_token == None:
                                break

                            # forced decoding
                            pred_token_forced = predict_token(encoder_output,
                                ys_gold, src_mask, trg_mask, model)
                            if pred_token_forced != gold_trg_token:
                                df.loc["incorrect predictions","gold history"] += 1
                            else:
                                df.loc["correct predictions","gold history"] += 1

                            # greedy decoding
                            pred_token_greedy = predict_token(encoder_output,
                                ys, src_mask, trg_mask, model)
                            if pred_token_greedy != gold_trg_token:
                                df.loc["incorrect predictions", "greedy history"] += 1
                            else:
                                df.loc["correct predictions", "greedy history"] += 1

                            if pred_token_forced != pred_token_greedy:
                                if pred_token_forced == gold_trg_token:
                                    only_gold += 1
                                if pred_token_greedy == gold_trg_token:
                                    only_predicted += 1

                            # history with only one mistake in the last token
                            pred_token_last = predict_token(encoder_output,
                                ys_last, src_mask, trg_mask, model)
                            if pred_token_last != gold_trg_token:
                                df.loc["incorrect predictions", "last token mistake"] += 1
                            else:
                                df.loc["correct predictions", "last token mistake"] += 1

                            # history with only one predicted mistake
                            for i in range(10):


                            pred_token_1mistake = predict_token(encoder_output,
                                ys_1mistake, src_mask, trg_mask, model)
                            if pred_token_1mistake != gold_trg_token:
                                one_mistake_incorrect[i] += 1
                                df.loc["incorrect predictions", "1 greedy mistake"] += 1
                            else:
                                one_mistake_correct[i] += 1
                                df.loc["correct predictions", "1 greedy mistake"] += 1

                            # beam search decoding
                            """pred_token_bs = predict_token(encoder_output,
                                ys_bs, src_mask, trg_mask, model)
                            if pred_token_bs != gold_trg_token:
                                df.loc["1 predicted mistake","incorrect predictions"] += 1
                            else:
                                df.loc["1 predicted mistake","correct predictions"] += 1"""

                            ys = torch.cat([ys, IntTensor([[pred_token_greedy]])], dim=1)
                            ys_gold = torch.cat([ys_gold, IntTensor([[gold_trg_token]])], dim=1)
                            pred_wrong_token = predict_wrong_token(gold_trg_token,
                                encoder_output, ys, src_mask, trg_mask, model)
                            ys_last = torch.cat([ys_gold, IntTensor([[pred_wrong_token]])], dim=1)
                            ys_1mistake = history_one_mistake(ys_gold, ys)
                            if bs_trg_token:
                                ys_bs = torch.cat([ys_bs, IntTensor([[bs_trg_token]])], dim=1)

    return df, only_gold, only_predicted

if __name__ == "__main__":

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    parser = argparse.ArgumentParser(description='Compares the mistakes \
    when using forced decoding versus greedy decoding.')
    parser.add_argument("source_corpus", help="path to the source corpus")
    parser.add_argument("target_corpus", help="path to the target corpus")
    parser.add_argument("predicted_corpus", help="path to the corpus translated \
    by the system using beam search")
    args = parser.parse_args()

    model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
    max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]


    df, only_gold, only_predicted = mistake_stats(args.source_corpus,
            args.target_corpus, args.predicted_corpus, model, max_output_length)

    print(df.to_string())
    print(f"\nNumber of tokens that are correctly predicted with forced decoding but"
          f" not with greedy decoding: {only_gold}.")
    print(f"\nNumber of tokens that are correctly predicted with greedy decoding but"
          f" not with forced decoding: {only_predicted}.\n")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
