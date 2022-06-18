'''
Creates and saves a matrix of size number of tokens x n
with the id of the n most probable tokens for each decision
And another matrix of same shape with the log probability for each of them
'''
N = 1000

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
import json

datetime_obj = datetime.datetime.now()
print(datetime_obj.time())

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]
bos_index = model.bos_index
eos_index = model.eos_index

f = open("/home/lina/Desktop/Stage/Modified_data/newstest2014.clean.bpe.fra")

# array of size number of tokens x n
res_tokens = []
res_lprob = []
for s, sentence in enumerate(f):

    print("translating sentence ", s)

    encoder_output = encode_sentence(sentence.split(), model)

    src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

    ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
    trg_mask = src_mask.new_ones([1, 1, 1])

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
        #print(to_tokens(pred_trg_token, model))

        n_best_tokens = []
        n_best_probabilities = []
        for j in range(N):
            n_best_tokens.append(int(log_probas.argmax()))
            n_best_probabilities.append(float(log_probas.max()))
            log_probas[0][log_probas.argmax()] = float('-inf')
        res_tokens.append(n_best_tokens)
        res_lprob.append(n_best_probabilities)

        if(pred_trg_token == eos_index):
            break

f.close()

with open("/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/nbest_tokens.newstest2014.json", 'w') as outfile:
    json.dump(res_tokens, outfile)

with open("/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/nbest_lprobs.newstest2014.json", 'w') as outfile:
    json.dump(res_lprob, outfile)

print(datetime_obj.time())
