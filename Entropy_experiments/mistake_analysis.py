import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

itertools.zip()

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
    df = pd.DataFrame([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    index=["correct_predictions", "incorrect_predictions"],
                    columns=["gold history", "greedy history", "beam search history",
                        "1 greedy mistake", "1 beam search mistake", "last token mistake"])

    one_greedy_correct = np.zeros(10)
    one_greedy_incorrect = np.zeros(10)
    one_bs_correct = np.zeros(10)
    one_bs_incorrect = np.zeros(10)
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

                        for gold_trg_token, bs_trg_token in itertools.zip_longest(gold_target, pred_target):
                            if gold_trg_token == None:
                                break

                            # forced decoding
                            pred_token_forced = predict_token(encoder_output,
                                ys_gold, src_mask, trg_mask, model)
                            if pred_token_forced != gold_trg_token:
                                df.loc["incorrect_predictions","gold history"] += 1
                            else:
                                df.loc["correct_predictions","gold history"] += 1

                            # greedy decoding
                            pred_token_greedy = predict_token(encoder_output,
                                ys, src_mask, trg_mask, model)
                            if pred_token_greedy != gold_trg_token:
                                df.loc["incorrect_predictions", "greedy history"] += 1
                            else:
                                df.loc["correct_predictions", "greedy history"] += 1

                            if pred_token_forced != pred_token_greedy:
                                if pred_token_forced == gold_trg_token:
                                    only_gold += 1
                                if pred_token_greedy == gold_trg_token:
                                    only_predicted += 1

                            # history with only one mistake in the last token
                            pred_token_last = predict_token(encoder_output,
                                ys_last, src_mask, trg_mask, model)
                            if pred_token_last != gold_trg_token:
                                df.loc["incorrect_predictions", "last token mistake"] += 1
                            else:
                                df.loc["correct_predictions", "last token mistake"] += 1

                            # history with only one greedy mistake
                            for i in range(10):
                                ys_1mistake = history_one_mistake(ys_gold, ys)
                                pred_token_1mistake = predict_token(encoder_output,
                                    ys_1mistake, src_mask, trg_mask, model)
                                if pred_token_1mistake != gold_trg_token:
                                    one_greedy_incorrect[i] += 1
                                else:
                                    one_greedy_correct[i] += 1

                            # beam search decoding
                            if ys_bs != None:
                                pred_token_bs = predict_token(encoder_output,
                                    ys_bs, src_mask, trg_mask, model)
                                if pred_token_bs != gold_trg_token:
                                    df.loc["incorrect_predictions", "beam search history"] += 1
                                else:
                                    df.loc["correct_predictions", "beam search history"] += 1

                            # history with only one beam search mistake
                            if ys_bs != None:
                                for i in range(10):
                                    ys_1mistake = history_one_mistake(ys_gold, ys_bs)
                                    pred_token_1mistake = predict_token(encoder_output,
                                        ys_1mistake, src_mask, trg_mask, model)
                                    if pred_token_1mistake != gold_trg_token:
                                        one_bs_incorrect[i] += 1
                                    else:
                                        one_bs_correct[i] += 1

                            ys = torch.cat([ys, IntTensor([[pred_token_greedy]])], dim=1)
                            ys_gold = torch.cat([ys_gold, IntTensor([[gold_trg_token]])], dim=1)
                            pred_wrong_token = predict_wrong_token(gold_trg_token,
                                encoder_output, ys, src_mask, trg_mask, model)
                            ys_last = torch.cat([ys_gold, IntTensor([[pred_wrong_token]])], dim=1)
                            if bs_trg_token:
                                ys_bs = torch.cat([ys_bs, IntTensor([[bs_trg_token]])], dim=1)
                            else:
                                ys_bs = None

    df.loc["incorrect_predictions", "1 greedy mistake"] = one_greedy_incorrect.mean()
    df.loc["correct_predictions", "1 greedy mistake"] = one_greedy_correct.mean()
    df.loc["incorrect_predictions", "1 beam search mistake"] = one_bs_incorrect.mean()
    df.loc["correct_predictions", "1 beam search mistake"] = one_bs_correct.mean()

    df = df.append(df.apply(
        lambda x: x.correct_predictions * 100 / (x.correct_predictions + x.incorrect_predictions),
        axis=0).rename("percentage correct"))
    df = pd.concat([df, pd.Series(["-", "-", "-", "-", "-", "-"], name="standard deviation")], ignore_index=True)
    print(df)
    df.loc["standard deviation", "1 greedy mistake"] = one_greedy_correct.std() \
        * 100 / (df.loc["correct_predictions", "1 greedy mistake"] + \
        df.loc["incorrect_predictions", "1 greedy mistake"])
    df.loc["standard deviation", "1 beam search mistake"] = one_bs_correct.std() \
        * 100 / (df.loc["correct_predictions", "1 beam search mistake"] + \
        df.loc["incorrect_predictions", "1 beam search mistake"])

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
    #args = parser.parse_args()
    args = parser.parse_args("/home/lina/Desktop/Stage/Modified_data/X-a-fini.small.bpe.fra /home/lina/Desktop/Stage/Modified_data/X-a-fini.small.bpe.eng /home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/tests/X-a-fini.small.bpe.eng".split())

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
