import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
N = 5
CORPUS = "X-a-fini.small"

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Predicting {N}-best translation hypothesis for sentences of {CORPUS} corpus with ancestral sampling")

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]

infile = open(f"/home/lina/Desktop/Stage/Modified_data/{CORPUS}.bpe.fra")
outfile = open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.ancestral.eng", "w")

for s, sentence in enumerate(infile):
    with Halo(text=f"Translating sentence {s}", spinner="dots"):
        src = encode_sentence(sentence.split(), model)
        for _ in range(N):
            outfile.write(to_tokens(ancestral_sampling(model, src, max_output_length), model) + "\n")

infile.close()
outfile.close()

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Finished predicting {N}-best translation hypothesis for sentences of {CORPUS} corpus with ancestral sampling")
