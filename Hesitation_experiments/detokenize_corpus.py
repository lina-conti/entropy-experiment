import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

parser = argparse.ArgumentParser(description='Detokenizes a corpus.')
parser.add_argument('corpus_path', help='path to the corpus to detokenize')
args = parser.parse_args()

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Detokenizing corpus.")

with Halo(text=f"Reading corpus", spinner="dots"):
    with open(args.corpus_path) as f:
        text_bpe = f.readlines()

with open(args.corpus_path, "w") as f:
    for s, sentence in enumerate(text_bpe):
        with Halo(text=f"Detokenizing sentence {s}.", spinner="dots"):
            tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
            f.write(tokenizer.decode(sentence.split()) + '\n')

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Finished detokenizing corpus.")
