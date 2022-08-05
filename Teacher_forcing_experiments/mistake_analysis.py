from utils_v2 import *

def analyse_sentence_mistakes(stats, tokens_list, sentence_src, sentence_trg, model, bos_index):
    encoder_output = encode_sentence(sentence_src, model)
    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])
    gold_target = [model.trg_vocab.lookup(token) for token in sentence_trg + [EOS_TOKEN]]
    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

    for gold_trg_token in gold_target:
        pred_trg_token, log_probs = predict_token(encoder_output, ys, \
            src_mask, trg_mask, model, return_log_probs=True)

        if pred_trg_token != gold_trg_token:
            stats["incorrect_nb"] += 1
            stats["prob_incorrect"] += log_probs[0][pred_trg_token].item()
            stats["entropy_incorrect"] += entropy(torch.exp(log_probs[0]).detach().cpu().numpy())
            tokens_list.append(\
                [model.trg_vocab.array_to_sentence([pred_trg_token])[0],\
                model.trg_vocab.array_to_sentence([gold_trg_token])[0],\
                log_probs[0][pred_trg_token].item(),\
                log_probs[0][gold_trg_token].item()])
        else:
            stats["correct_nb"] += 1
            stats["prob_correct"] += log_probs[0][pred_trg_token].item()
            stats["entropy_correct"] += entropy(torch.exp(log_probs[0]).detach().cpu().numpy())

        if TEACHER_FORCING:
            ys = torch.cat([ys, IntTensor([[gold_trg_token]])], dim=1)
        else:
            ys = torch.cat([ys, IntTensor([[pred_trg_token]])], dim=1)

def mistake_stats(src_corpus: str, trg_corpus: str, model: Model, config: Dict):

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

    tokenizer = build_tokenizer(config["data"])
    src_tokenizer = tokenizer[config["data"]["src"]["lang"]]
    trg_tokenizer = tokenizer[config["data"]["trg"]["lang"]]

    with open(src_corpus) as f_src:
        with open(trg_corpus) as f_trg:
            s = 0
            for sentence_src, sentence_trg in zip(f_src, f_trg):
                s += 1
                sentence_src = src_tokenizer(src_tokenizer.pre_process(sentence_src))
                sentence_trg = trg_tokenizer(trg_tokenizer.pre_process(sentence_trg))
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

    parser = argparse.ArgumentParser(description='Analyses the mistakes commited \
    using teacher forcing or not.')
    parser.add_argument("model_path", help="path to the config of the model to be used")
    parser.add_argument("source_corpus", help="path to the source corpus, tokenized with bpe")
    parser.add_argument("target_corpus", help="path to the gold target corpus, tokenized with bpe")
    parser.add_argument("-o", "--output_path", help="path to the csv file where results should be saved")
    parser.add_argument("-t", "--teacher_forcing", help="whether to use a gold\
    prefix (teacher forcing) or predicted prefix", action="store_true")
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    TEACHER_FORCING = args.teacher_forcing
    model = load_model(args.model_path)
    config = load_config(args.model_path)

    df_tokens, df_res = mistake_stats(args.source_corpus, args.target_corpus, model, config)

    print(df_res.to_string(), "\n")

    if args.output_path:
        with Halo(text=f"saving results", spinner="dots") as spinner:
            with open(args.output_path,'a') as f:
                f.truncate(0)
                f.write(f"teacher forcing = {TEACHER_FORCING}\n")
                for df in df_res, df_tokens:
                    df.to_csv(f)
                    f.write("\n")
        spinner.succeed(f"Results saved to {args.output_path}")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
