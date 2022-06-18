import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")

s = "▁spec ta cul aire ▁sau t ▁en ▁\" wing suit \" ▁au - des s us ▁de ▁bog ota"
target = "▁spec ta cul aire ▁sau t ▁en ▁\" wing suit \" ▁au - des s us ▁de ▁bog ota".split()

encoder_output = encode_sentence(s.split(), model)

src_mask = torch.tensor([[[True for _ in range(encoder_output.shape[1])]]])

target = [model.trg_vocab.stoi[token] for token in target + [EOS_TOKEN]]

bos_index = model.bos_index
eos_index = model.eos_index

ys = encoder_output.new_full([1, 1], bos_index, dtype=torch.long)
trg_mask = src_mask.new_ones([1, 1, 1])

res = []
for gold_trg_token in target:
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
        print(to_tokens(pred_trg_token, model))
        pred_trg_token = pred_trg_token.data.unsqueeze(-1)
        res.append({"predicted_token_idx": pred_trg_token.item(),
                    "predicted_log_proba": log_probas[0][pred_trg_token].item(),
                    "gold_token_idx": gold_trg_token,
                    "gold_log_proba": log_probas[0][gold_trg_token].item(),
                    "log_probas": log_probas[0].detach().cpu().numpy()
                    })

        ys = torch.cat([ys, IntTensor([[gold_trg_token]])], dim=1)
