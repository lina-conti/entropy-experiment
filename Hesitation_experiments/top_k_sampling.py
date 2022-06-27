import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]


s = "▁l ' athlète ▁a ▁terminé ▁son ▁travail ▁."
#t = "▁the ▁athlete ▁finished ▁his ▁work ▁."

src = encode_sentence(s.split(), model)

print(to_tokens(top_k_sampling(model, src, max_output_length, 10), model))
