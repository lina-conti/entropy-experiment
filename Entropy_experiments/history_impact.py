import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

def compute_percentages(df, one_greedy_correct, one_bs_correct):
    df_percentages = pd.concat([ \
        df.apply(lambda x: x.correct_predictions * 100 / (x.correct_predictions \
         + x.incorrect_predictions), axis=0).rename("percentage correct"), \
        pd.Series(["-", "-", "-", "-", "-", "-"], name="standard deviation", \
        index = df.columns)], axis=1)

    df_percentages = df_percentages.transpose()

    df_percentages.loc["standard deviation", "1 greedy mistake*"] = one_greedy_correct.std() \
        * 100 / (df.loc["correct_predictions", "1 greedy mistake*"] + \
        df.loc["incorrect_predictions", "1 greedy mistake*"])
    df_percentages.loc["standard deviation", "1 beam search mistake*"] = one_bs_correct.std() \
        * 100 / (df.loc["correct_predictions", "1 beam search mistake*"] + \
        df.loc["incorrect_predictions", "1 beam search mistake*"])

    return df_percentages

def mistake_stats(src_corpus: str, trg_corpus: str, pred_corpus: str, model: Model,
                  max_output_length: int) -> (pd.DataFrame, pd.DataFrame, int, int):

    bos_index = model.bos_index
    eos_index = model.eos_index
    only_gold = 0
    only_predicted = 0
    df = pd.DataFrame([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    index=["correct_predictions", "incorrect_predictions"],
                    columns=["gold history", "greedy history", "beam search history",
                        "1 greedy mistake*", "1 beam search mistake*", "last token mistake"],
                    dtype='object')

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

                        if s != 1:
                            print()
                            print("ys gold:\t", to_tokens(ys_gold[0], model))
                            print("ys greedy:\t", to_tokens(ys[0], model))
                            print("ys bs:    \t", to_tokens(ys_bs[0], model))
                            print()

                        ys = ys_gold = ys_last = ys_1mistake = ys_bs = encoder_output.new_full(
                                                    [1, 1], bos_index, dtype=torch.long)
                        trg_mask = src_mask.new_ones([1, 1, 1])

                        for gold_trg_token, bs_trg_token in itertools.zip_longest(gold_target, pred_target):
                            if gold_trg_token == None:
                                break

                            # gold history
                            pred_token_forced = predict_token(encoder_output,
                                ys_gold, src_mask, trg_mask, model)
                            if pred_token_forced != gold_trg_token:
                                df.loc["incorrect_predictions","gold history"] += 1
                            else:
                                df.loc["correct_predictions","gold history"] += 1

                            # greedy decoding
                            # TODO: adapt this part based on what is done when greedy ends
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
                            # TODO: adapt this part based on what is done when bs ends
                            if ys_bs != None:
                                pred_token_bs = predict_token(encoder_output,
                                    ys_bs, src_mask, trg_mask, model)
                                if pred_token_bs != gold_trg_token:
                                    df.loc["incorrect_predictions", "beam search history"] += 1
                                else:
                                    df.loc["correct_predictions", "beam search history"] += 1
                            else:
                                df.loc["incorrect_predictions", "beam search history"] += 1

                            # history with only one beam search mistake
                            for i in range(10):
                                if ys_bs.size() == ys_gold.size():
                                    ys_1mistake = history_one_mistake(ys_gold, ys_bs)
                                    pred_token_1mistake = predict_token(encoder_output,
                                        ys_1mistake, src_mask, trg_mask, model)
                                    if pred_token_1mistake != gold_trg_token:
                                        one_bs_incorrect[i] += 1
                                    else:
                                        one_bs_correct[i] += 1
                                else:
                                    one_bs_incorrect[i] += 1

                            # WIP: careful, this way all histories don't have
                            # the same size and some cannot be used
                            # TODO: decide how to handle one mistake max and
                            # implement it
                            if ys[0][-1] != eos_index:
                                ys = torch.cat([ys, IntTensor([[pred_token_greedy]])], dim=1)
                            ys_gold = torch.cat([ys_gold, IntTensor([[gold_trg_token]])], dim=1)
                            pred_wrong_token = predict_wrong_token(gold_trg_token,
                                encoder_output, ys, src_mask, trg_mask, model)
                            ys_last = torch.cat([ys_gold, IntTensor([[pred_wrong_token]])], dim=1)
                            if bs_trg_token:
                                ys_bs = torch.cat([ys_bs, IntTensor([[bs_trg_token]])], dim=1)


    df.loc["incorrect_predictions", "1 greedy mistake*"] = one_greedy_incorrect.mean()
    df.loc["correct_predictions", "1 greedy mistake*"] = one_greedy_correct.mean()
    df.loc["incorrect_predictions", "1 beam search mistake*"] = one_bs_incorrect.mean()
    df.loc["correct_predictions", "1 beam search mistake*"] = one_bs_correct.mean()

    df_percentages = compute_percentages(df, one_greedy_correct, one_bs_correct)

    return df, df_percentages, only_gold, only_predicted

if __name__ == "__main__":

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    parser = argparse.ArgumentParser(description='Compares the mistakes \
    when using forced decoding versus greedy decoding.')
    parser.add_argument("source_corpus", help="path to the source corpus, tokenized with bpe")
    parser.add_argument("target_corpus", help="path to the target corpus, tokenized with bpe")
    parser.add_argument("predicted_corpus", help="path to the corpus translated \
    by the system using beam search, tokenized with bpe")
    args = parser.parse_args()

    model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
    max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]


    df, df_percentages, only_gold, only_predicted = mistake_stats(args.source_corpus,
            args.target_corpus, args.predicted_corpus, model, max_output_length)

    print(df.to_string(), "\n")
    print(df_percentages.to_string())
    print(f"\nNumber of tokens that are correctly predicted with forced decoding but"
          f" not with greedy decoding: {only_gold}.")
    print(f"\nNumber of tokens that are correctly predicted with greedy decoding but"
          f" not with forced decoding: {only_predicted}.\n")

    print("*For these columns, the experiment was repeated 10 times, the results shown are the averages over all 10 experiments.\n")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
