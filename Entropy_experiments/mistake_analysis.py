import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

def mistake_stats(src_corpus: str, trg_corpus: str, model: Model,
                  max_output_length: int) -> (pd.DataFrame, int, int):

    bos_index = model.bos_index
    only_gold = 0
    only_predicted = 0
    df = pd.DataFrame([[0, 0], [0, 0]],
                    index=["correct predictions", "incorrect predictions"],
                    columns=["gold history", "predicted history"])

    with open(src_corpus) as f_src:
        with open(trg_corpus) as f_trg:
            s = 0
            for sentence_fr, sentence_en in zip(f_src, f_trg):
                s += 1
                with Halo(text=f"translating sentence {s}", spinner="dots"):

                    encoder_output = encode_sentence(sentence_fr.strip().split(), model)

                    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

                    target = [model.trg_vocab.stoi[token] for token in sentence_en.strip().split() + [EOS_TOKEN]]

                    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
                    ys_gold = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
                    trg_mask = src_mask.new_ones([1, 1, 1])

                    for i, gold_trg_token in enumerate(target):
                        # forced decoding
                        model.eval()
                        with torch.no_grad():
                            logits, _, _, _ = model(
                                return_type="decode",
                                trg_input=ys_gold,
                                encoder_output=encoder_output,
                                encoder_hidden=None,
                                src_mask=src_mask,
                                unroll_steps=None,
                                decoder_hidden=None,
                                trg_mask=trg_mask
                            )
                        logits = logits[:, -1]
                        max_value, pred_token_forced = torch.max(logits, dim=1)
                        pred_token_forced = int(pred_token_forced.data.unsqueeze(-1))
                        if pred_token_forced != gold_trg_token:
                            df["gold history"]["incorrect predictions"] += 1
                        else:
                            df["gold history"]["correct predictions"] += 1

                        # greedy_decoding
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
                        max_value, pred_token_greedy = torch.max(logits, dim=1)
                        pred_token_greedy = int(pred_token_greedy.data.unsqueeze(-1))
                        if pred_token_greedy != gold_trg_token:
                            df["predicted history"]["incorrect predictions"] += 1
                        else:
                            df["predicted history"]["correct predictions"] += 1

                        if pred_token_forced != pred_token_greedy:
                            if pred_token_forced == gold_trg_token:
                                only_gold += 1
                            if pred_token_greedy == gold_trg_token:
                                only_predicted += 1

                        ys = torch.cat([ys, IntTensor([[pred_token_greedy]])], dim=1)
                        ys_gold = torch.cat([ys_gold, IntTensor([[gold_trg_token]])], dim=1)

    return df, only_gold, only_predicted

if __name__ == "__main__":

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started translating the corpus.")

    parser = argparse.ArgumentParser(description='Compares the mistakes \
    when using forced decoding versus greedy decoding.')
    parser.add_argument("source_corpus", help="path to the source corpus")
    parser.add_argument("target_corpus", help="path to the target corpus")
    args = parser.parse_args()

    model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
    max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]


    df, only_gold, only_predicted = mistake_stats(args.source_corpus,
                            args.target_corpus, model, max_output_length)

    print(df)
    print(f"\nNumber of tokens that are correctly predicted with forced decoding but"
          f" not with greedy decoding: {only_gold}.")
    print(f"\nNumber of tokens that are correctly predicted with greedy decoding but"
          f" not with forced decoding: {only_predicted}.\n")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished analysing the translations.")
