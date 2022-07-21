import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

def analyse_sentence_mistakes(stats, tokens_list, sentence_src, sentence_trg, model, bos_index):
    encoder_output = encode_sentence(sentence_src.strip().split(), model)
    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])
    gold_target = [model.trg_vocab.stoi[token] for token in sentence_trg.strip().split() + [EOS_TOKEN]]
    ys_gold = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

    for gold_trg_token in gold_target:
        pred_trg_token, log_probs = predict_token(encoder_output, ys_gold, \
            src_mask, trg_mask, model, return_log_probs=True)

        if pred_trg_token != gold_trg_token:
            stats["incorrect_nb"] += 1
            stats["prob_incorrect"] += log_probs[0][pred_trg_token].item()
            stats["entropy_incorrect"] += entropy(torch.exp(log_probs[0]).detach().cpu().numpy())
            tokens_list.append(\
                [model.trg_vocab.itos[pred_trg_token],\
                model.trg_vocab.itos[gold_trg_token],\
                log_probs[0][pred_trg_token].item(),\
                log_probs[0][gold_trg_token].item()])
        else:
            stats["correct_nb"] += 1
            stats["prob_correct"] += log_probs[0][pred_trg_token].item()
            stats["entropy_correct"] += entropy(torch.exp(log_probs[0]).detach().cpu().numpy())

        ys_gold = torch.cat([ys_gold, IntTensor([[gold_trg_token]])], dim=1)

def mistake_stats(src_corpus: str, trg_corpus: str, model: Model):

    bos_index = model.bos_index

    tokens_list = []
    stats = {
        "prob_correct": 0,
        "prob_incorrect": 0,
        "entropy_correct": 0,
        "entropy_incorrect": 0,
        "correct_nb": 0,
        "incorrect_nb": 0
    }

    with open(src_corpus) as f_src:
        with open(trg_corpus) as f_trg:
            s = 0
            for sentence_src, sentence_trg in zip(f_src, f_trg):
                s += 1
                with Halo(text=f"translating sentence {s}", spinner="dots"):
                    analyse_sentence_mistakes(stats, tokens_list, sentence_src, sentence_trg, model, bos_index)

    df_tokens = pd.DataFrame(tokens_list,
                    columns=["Predicted token", "Gold token", \
                    "Predicted token log probability", "Gold token log probability"])

    df_res = pd.DataFrame([\
        [(df_tokens["Predicted token log probability"] - df_tokens["Gold token log probability"]).mean()],\
         [stats["prob_correct"] / stats["correct_nb"]],\
         [stats["prob_incorrect"] / stats["incorrect_nb"]],\
         [stats["entropy_correct"] / stats["correct_nb"]],\
         [stats["entropy_incorrect"] / stats["incorrect_nb"]]],
        index=["Average difference between predicted token and gold token log probabilities",\
         "Average log probability of the predicted token in a correct decision",\
         "Average log probability of the predicted token in an incorrect decision",\
         "Average entropy in a correct decision",\
         "Average entropy in an incorrect decision"],
        columns=["Value"],
        dtype='object')

    return df_tokens, df_res

if __name__ == "__main__":

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    parser = argparse.ArgumentParser(description='Ananlyses the mistakes commited \
    despite using forced decoding with gold target input.')
    parser.add_argument("source_corpus", help="path to the source corpus, tokenized with bpe")
    parser.add_argument("target_corpus", help="path to the gold target corpus, tokenized with bpe")
    parser.add_argument("-o", "--output_path", help="path to the csv file where results should be saved")
    args = parser.parse_args()

    model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")

    df_tokens, df_res = mistake_stats(args.source_corpus, args.target_corpus, model)

    print(df_res.to_string(), "\n")

    if args.output_path:
        with Halo(text=f"saving results", spinner="dots") as spinner:
            with open(args.output_path,'a') as f:
                f.truncate(0)
                for df in df_res, df_tokens:
                    df.to_csv(f)
                    f.write("\n")
        spinner.succeed(f"Results saved to {args.output_path}")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
