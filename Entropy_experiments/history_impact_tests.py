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

    ys_gold = ys = ys_bs = encoder_output.new_full(
                                [1, 1], bos_index, dtype=torch.long)
    bs_finished = False

    for gold_token, original_token_bs in itertools.zip_longest(gold_target, pred_target):
        if gold_token == None:
            break

        bs_finished = original_token_bs == None

        # gold history
        token_forced = predict_token(encoder_output, ys_gold, src_mask, trg_mask, model)
        counters_dict["correct_gold"] += token_forced == gold_token
        counters_dict["incorrect_gold"] += token_forced != gold_token

        # greedy beam search decoding
        if not bs_finished:
            predicted_token_bs = predict_token(encoder_output, ys_bs, src_mask, trg_mask, model)
            counters_dict["different_tokens"] += original_token_bs != predicted_token_bs
            counters_dict["both_wrong"] += (original_token_bs != predicted_token_bs) \
             and (original_token_bs != gold_token) and (gold_token != predicted_token_bs)
        counters_dict["correct_greedy_bs"] += not bs_finished and predicted_token_bs == gold_token
        counters_dict["incorrect_greedy_bs"] += bs_finished or predicted_token_bs != gold_token

        # beam search decoding
        counters_dict["correct_bs"] += not bs_finished and original_token_bs == gold_token
        counters_dict["incorrect_bs"] += bs_finished or original_token_bs != gold_token

        ys_gold = torch.cat([ys_gold, IntTensor([[gold_token]])], dim=1)
        if not bs_finished:
            ys_bs = torch.cat([ys_bs, IntTensor([[original_token_bs]])], dim=1)

def compute_results(counters_dict):
    data = {'gold history': [counters_dict['correct_gold'], counters_dict['incorrect_gold']],
            'beam search history': [counters_dict['correct_greedy_bs'], counters_dict['incorrect_greedy_bs']],
            'beam search': [counters_dict['correct_bs'], counters_dict['incorrect_bs']]}

    df_counts = pd.DataFrame(data, index=['correct_predictions',
                                          'incorrect_predictions'])

    df_percentages = df_counts.apply(lambda x: round(x.correct_predictions * 100\
        / (x.correct_predictions + x.incorrect_predictions), 2), axis=0)\
        .rename("percentage correct")

    print(f"\nNumber of differences between beam search tokens and tokens predicted"
          f" with beam search prefix: {counters_dict['different_tokens']}")

    print(f"Number of cases where they are different, but both wrong: {counters_dict['both_wrong']}\n")

    return df_counts.T, df_percentages

def history_impact(src_corpus: str, trg_corpus: str, pred_corpus: str, model: Model,
                  max_output_length: int):

    bos_index = model.bos_index
    eos_index = model.eos_index

    counters_dict = {
        "correct_gold": 0, "incorrect_gold": 0,
        "correct_greedy_bs": 0, "incorrect_greedy_bs": 0,
        "correct_bs": 0, "incorrect_bs": 0,
        "different_tokens": 0, "both_wrong":0
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
    return df_counts, df_percentages

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analyses the effect of changes \
    in the target history on the number of correctly predicted tokens.')
    parser.add_argument("source_corpus", help="path to the source corpus, tokenized with bpe")
    parser.add_argument("target_corpus", help="path to the gold target corpus, tokenized with bpe")
    parser.add_argument("predicted_corpus", help="path to the corpus translated \
    by the system using beam search, tokenized with bpe")
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
    max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]

    df_counts, df_percentages = history_impact(
        args.source_corpus, args.target_corpus, args.predicted_corpus, model, max_output_length)

    print(df_counts.to_string(), "\n")
    print(df_percentages.to_string(), "\n")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
