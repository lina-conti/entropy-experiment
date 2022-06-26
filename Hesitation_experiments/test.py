import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
from lingpy.align.multiple import mult_align
import sentencepiece as spm
import stanza
import difflib
from pprint import pprint

N = 5
CORPUS = "newstest2014.pred"
POS_TAGS = False

def get_modifications(sentenceA, sentenceB):
    '''
    takes as input two lists of words and returns a dictionary
    with deletions, insertions and substition necesaary for going from
    the first to the second
    '''
    res = {}
    matcher = difflib.SequenceMatcher(None, sentenceA, sentenceB)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            res[tag] = sentenceA[i1:i2], sentenceB[j1:j2]
    return res

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
diff = difflib.Differ()

hypothesis = []
with open(f"/home/lina/Desktop/Stage/Modified_data/{CORPUS}.eng", 'r') as infile:
    for i, line in enumerate(infile):
        hypothesis.append(tokenizer.decode(line.split()))
        if (i+1)%N == 0:
            #aligned_hyp = mult_align(hypothesis)
            """for hyp in aligned_hyp:
                print("\t".join(w for w in hyp))"""
            for j in range(1, len(hypothesis)):
                print(hypothesis[0])
                print(hypothesis[j])
                #print("\t".join(w for w in list(diff.compare(hypothesis[0].split(), hypothesis[j].split()))))
                print(get_modifications(hypothesis[0].split(), hypothesis[j].split()))
                print()
            print()
            hypothesis = []
