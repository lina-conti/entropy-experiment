from utils import *

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]
bos_index = model.bos_index
eos_index = model.eos_index

file_fr = open("/home/lina/Desktop/Stage/Modified_data/X-a-fini.gold.bpe.fra")
file_en = open("/home/lina/Desktop/Stage/Modified_data/X-a-fini.gold.bpe.eng")

#res_df = pd.DataFrame(columns=("token_position", "sentence_length", "log_probas", "entropy"))
res_df = pd.DataFrame(columns=("token_position", "sentence_length", "entropy"))
token_counter = 0
s = 0
for sentence_fr, sentence_en in zip(file_fr, file_en):
    s += 1

    print("translating sentence ", s)

    encoder_output = encode_sentence(sentence_fr.strip().split(), model)

    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

    target = [model.trg_vocab.stoi[token] for token in sentence_en.strip().split() + [EOS_TOKEN]]

    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

    bos_id = token_counter
    for i, gold_trg_token in enumerate(target):
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
        probas = softmax(logits)

        res_df.loc[token_counter] = pd.Series({"token_position": i,
                    "sentence_length": -1,
                    "entropy": entropy(probas[0].detach().cpu().numpy())
                    })

        ys = torch.cat([ys, IntTensor([[gold_trg_token]])], dim=1)

        token_counter += 1

    res_df["sentence_length"][bos_id:bos_id + i + 1] = i

file_fr.close()
file_en.close()

res_df.to_pickle('/home/lina/Desktop/Stage/Experience_entropie/results/entropies_X-a-fini_forced.pickle')
