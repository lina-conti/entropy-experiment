import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

def history_one_mistake(gold_history: Tensor, mistakes_history: Tensor) -> Tensor:
    rng = np.random.default_rng()
    indices = list(range(mistakes_history.size()[1] -2))
    i = rng.choice(indices) + 1
    new_history = gold_history.detach().clone()
    new_history[0][i] = mistakes_history[0][i]
    return new_history

def analyse_sentence(counters_dict, sentence_src, sentence_trg, sentence_pred, model, bos_index, eos_index):
    encoder_output = encode_sentence(sentence_src.strip().split(), model)
    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])
    gold_target = [model.trg_vocab.stoi[token] for token in sentence_trg.strip().split() + [EOS_TOKEN]]
    pred_target = [model.trg_vocab.stoi[token] for token in sentence_pred.strip().split() + [EOS_TOKEN]]
    trg_mask = src_mask.new_ones([1, 1, 1])

    ys_gold = ys = ys_bs = ys_wrong = encoder_output.new_full(
                                [1, 1], bos_index, dtype=torch.long)
    greedy_finished = False
    bs_finished = False

    for gold_token, original_token_bs in itertools.zip_longest(gold_target, pred_target):
        if gold_token == None:
            break

        bs_finished = original_token_bs == None

        # gold history
        token_forced = predict_token(encoder_output, ys_gold, src_mask, trg_mask, model)
        counters_dict["correct_gold"] += token_forced == gold_token
        counters_dict["incorrect_gold"] += token_forced != gold_token

        # greedy decoding
        if not greedy_finished:
            token_greedy = predict_token(encoder_output, ys, src_mask, trg_mask, model)
        counters_dict["correct_greedy"] += not greedy_finished and token_greedy == gold_token
        counters_dict["incorrect_greedy"] += greedy_finished or token_greedy != gold_token
        counters_dict["only_predicted"] += \
            (token_forced != gold_token) and (not greedy_finished and token_greedy == gold_token)
        counters_dict["only_gold"] += \
            (token_forced == gold_token) and (greedy_finished or token_greedy != gold_token)

        # beam search decoding
        if not bs_finished:
            predicted_token_bs = predict_token(encoder_output, ys_bs, src_mask, trg_mask, model)
        counters_dict["correct_bs"] += not bs_finished and predicted_token_bs == gold_token
        counters_dict["incorrect_bs"] += bs_finished or predicted_token_bs != gold_token

        # history with one mistake in the last token
        if ys_gold.size()[1] > 1:
            token_last = predict_token(encoder_output, ys_last, src_mask, trg_mask, model)
            counters_dict["correct_last"] += token_last == gold_token
            counters_dict["incorrect_last"] += token_last != gold_token

        # history with one mistake in another part of the sentence
        if ys_gold.size()[1] > 2:
            for i in range(10):
                ys_1mistake = history_one_mistake(ys_gold, ys_wrong)
                token_1mistake = predict_token(encoder_output, ys_1mistake, src_mask, trg_mask, model)
                counters_dict["correct_1mistake"][i] += token_1mistake == gold_token
                counters_dict["incorrect_1mistake"][i] += token_1mistake != gold_token

        ys_gold = torch.cat([ys_gold, IntTensor([[gold_token]])], dim=1)
        greedy_finished = token_greedy == eos_index
        if not greedy_finished:
            ys = torch.cat([ys, IntTensor([[token_greedy]])], dim=1)
        if not bs_finished:
            ys_bs = torch.cat([ys_bs, IntTensor([[original_token_bs]])], dim=1)
        wrong_token = predict_wrong_token(gold_token, encoder_output, ys, src_mask, trg_mask, model)
        ys_last = torch.cat([ys_gold, IntTensor([[wrong_token]])], dim=1)
        ys_wrong = torch.cat([ys_wrong, IntTensor([[wrong_token]])], dim=1)

def compute_results(counters_dict):
    data = {'gold history': [counters_dict['correct_gold'], counters_dict['incorrect_gold'], '-'],
            'greedy history': [counters_dict['correct_greedy'], counters_dict['incorrect_greedy'], '-'],
            'beam search history': [counters_dict['correct_bs'], counters_dict['incorrect_bs'], '-'],
            'last token mistake': [counters_dict['correct_last'], counters_dict['incorrect_last'], '-'],
            'one mistake': [counters_dict['correct_1mistake'].mean(), counters_dict['incorrect_1mistake'].mean(),
            f"{round(counters_dict['correct_1mistake'].std(), 2)}"]}

    print(f"\ncounters_dict['correct_1mistake']: {round(counters_dict['correct_1mistake'].std(), 2)}")
    print(f"counters_dict['incorrect_1mistake']: {round(counters_dict['incorrect_1mistake'].std(), 2)}")

    df_counts = pd.DataFrame(data, index=['correct_predictions',
                                          'incorrect_predictions',
                                          'standard_deviation'])

    df_percentages = pd.concat([ \
        df_counts.apply(lambda x: round(x.correct_predictions * 100 / (x.correct_predictions \
         + x.incorrect_predictions), 2), axis=0).rename("percentage correct"), \
        pd.Series(["-", "-", "-", "-", "-"], name="standard deviation", \
        index = df_counts.columns)], axis=1)

    percentages_1mist = np.zeros(10)
    for i in range(10):
        percentages_1mist[i] = counters_dict['correct_1mistake'][i] * 100 \
            / (counters_dict['correct_1mistake'][i] + counters_dict['incorrect_1mistake'][i])

    print(f"percentages_1mist.mean(): {round(percentages_1mist.mean(), 2)}")
    print(f'df_percentages.loc["one mistake", "percentage correct"]: {df_percentages.loc["one mistake", "percentage correct"]}\n')

    df_percentages.loc["one mistake", "percentage correct"] = round(percentages_1mist.mean(), 2)
    df_percentages.loc["one mistake", "standard deviation"] = round(percentages_1mist.std(), 2)

    return df_counts.T, df_percentages

def history_impact(src_corpus: str, trg_corpus: str, pred_corpus: str, model: Model,
                  max_output_length: int):

    bos_index = model.bos_index
    eos_index = model.eos_index

    counters_dict = {
        "only_gold": 0, "only_predicted": 0,
        "correct_gold": 0, "incorrect_gold": 0,
        "correct_greedy": 0, "incorrect_greedy": 0,
        "correct_bs": 0, "incorrect_bs": 0,
        "correct_last": 0, "incorrect_last": 0,
        "correct_1mistake": np.zeros(10), "incorrect_1mistake": np.zeros(10)
    }

    with open(src_corpus) as f_src:
        with open(trg_corpus) as f_trg:
            with open(pred_corpus) as f_pred:
                s = 0
                for sentence_src, sentence_trg, sentence_pred in zip(f_src, f_trg, f_pred):
                    s += 1
                    with Halo(text=f"translating sentence {s}", spinner="dots"):
                        analyse_sentence(counters_dict, sentence_src, sentence_trg,
                                        sentence_pred, model, bos_index, eos_index)

    df_counts, df_percentages = compute_results(counters_dict)
    return df_counts, df_percentages, counters_dict["only_gold"], counters_dict["only_predicted"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analyses the effect of changes \
    in the target history on the number of correctly predicted tokens.')
    parser.add_argument("source_corpus", help="path to the source corpus, tokenized with bpe")
    parser.add_argument("target_corpus", help="path to the gold target corpus, tokenized with bpe")
    parser.add_argument("predicted_corpus", help="path to the corpus translated \
    by the system using beam search, tokenized with bpe")
    parser.add_argument("-o", "--output_path", help="path to the csv file where results should be saved")
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
    max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]

    df_counts, df_percentages, only_gold, only_predicted = history_impact(
        args.source_corpus, args.target_corpus, args.predicted_corpus, model, max_output_length)

    print(df_counts.to_string(), "\n")
    print(df_percentages.to_string(), "\n")
    print(f"Number of tokens that are correctly predicted with forced decoding but"
          f" not with greedy decoding: {only_gold}.")
    print(f"\nNumber of tokens that are correctly predicted with greedy decoding but"
          f" not with forced decoding: {only_predicted}.\n")

    if args.output_path:
        with Halo(text=f"Saving results", spinner="dots") as spinner:
            with open(args.output_path,'a') as f:
                f.truncate(0)
                for df in df_counts, df_percentages:
                    df.to_csv(f)
                    f.write("\n")
                f.write(f"Number of tokens that are correctly predicted with "
                    f"forced decoding but not with greedy decoding, {only_gold}\n")
                f.write(f"\nNumber of tokens that are correctly predicted with "
                    f"greedy decoding but not with forced decoding, {only_predicted}")
        spinner.succeed(f"Results saved to {args.output_path}")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
