import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
import stanza

parser = argparse.ArgumentParser(description='Translates a corpus of sentences \
to POS tags.')
parser.add_argument('corpus_path', help='corpus to translate')
parser.add_argument('output_path', help='where to write the generated text to')
args = parser.parse_args()

with open(args.corpus_path) as infile:
    with open(args.output_path, "w") as outfile:
        text_bpe = infile.read()
        tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
        text = tokenizer.decode(text_bpe.split())
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
        doc = nlp(text)
        for sentence in doc.sentences:
            outfile.write(" ".join(word.upos for word in sentence.words) + "\n")
