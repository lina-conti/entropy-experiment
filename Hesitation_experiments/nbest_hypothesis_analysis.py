"""
Compares the N best hypothesis predicted by the system:
- hypothesis are aligned and written to a file
- hypothesis can also be annotated for POS (optional)
"""
N = 5
CORPUS = "X-a-fini.small.pred"
POS_TAGS = False

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
from lingpy.align.multiple import mult_align
import sentencepiece as spm
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')

hypothesis = []
with open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.eng", 'r') as infile:
    with open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.aligned.eng", 'w') as outfile:
        for i, line in enumerate(infile):
            hypothesis.append(line)
            if (i+1)%N == 0:
                aligned_hyp = mult_align(hypothesis)
                for j in range(len(hypothesis)):
                    outfile.write("\t".join(token for token in aligned_hyp[j]))
                    if POS_TAGS:
                        doc = nlp(tokenizer.decode(hypothesis[j].split()))
                        outfile.write("\t".join(word.upos for sent in doc.sentences for word in sent.words))
                        outfile.write("\n")
                hypothesis = []
                outfile.write("\n")
