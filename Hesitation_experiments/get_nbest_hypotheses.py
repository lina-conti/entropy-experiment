import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

parser = argparse.ArgumentParser(description='Predicts the n-best translation \
hypothesis for sentences in a given corpus with a given decoding strategy.')
parser.add_argument('n', help='number of translation hypothesis to generate \
for each sentence', type=int)
parser.add_argument('corpus', help='corpus to translate')
parser.add_argument('decoding_strategy', choices=["ancestral_sampling", "top_k"], help='decoding strategy to use')
parser.add_argument('--k', help='value of k if top-k sampling decoding strategy is used', type=int)

args = parser.parse_args()

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Predicting {args.n}-best translation hypotheses \
for sentences of {args.corpus} corpus with {args.decoding_strategy} decoding strategy.")

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]

infile = open(f"/home/lina/Desktop/Stage/Modified_data/{args.corpus}.gold.bpe.fra")
outfile = open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{args.corpus}.{args.decoding_strategy}.eng", "w")

for s, sentence in enumerate(infile):
    hypotheses = []
    with Halo(text=f"Translating sentence {s}", spinner="dots"):
        src = encode_sentence(sentence.split(), model)
        for _ in range(args.n):
            while True:
                if args.decoding_strategy == 'ancestral_sampling':
                    hyp = to_tokens(ancestral_sampling(model, src, max_output_length), model)
                if args.decoding_strategy == 'top_k':
                    if not args.k:
                        print('\nNo value was given for k.\nPlease see get_nbest_hypotheses.py -h for help.')
                        exit()
                    hyp = to_tokens(top_k_sampling(model, src, max_output_length, args.k), model)
                if hyp not in hypotheses:
                    hypotheses.append(hyp)
                    break

            outfile.write(hyp + "\n")

infile.close()
outfile.close()

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Finished predicting {args.n}-best translation hypotheses \
for sentences of {args.corpus} corpus with {args.decoding_strategy} decoding strategy.")
