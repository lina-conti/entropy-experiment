import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
from lingpy.align.multiple import mult_align
import sentencepiece as spm
import stanza
import difflib
from pprint import pprint

N = 5
CORPUS = "newstest2014.small"
POS_TAGS = False

def get_modifications(delta_list):
    '''
    takes as input a list obtained with Differ.compare and returns a dictionary
    with deleted words, inserted words and substituted words
    '''
    res = {'Substitutions': [], 'Insertions': [], 'Deletions': []}
    i = 0
    while i < len(delta_list):
        deletions = ''
        insertions = ''
        while i<len(delta_list) and delta_list[i].startswith('+'):
            insertions += delta_list[i][1:]
            i += 1
        if insertions:
            res['Insertions'].append(insertions)
            continue
        while i<len(delta_list) and delta_list[i].startswith('-'):
            deletions += delta_list[i][1:]
            i += 1
        if deletions:
            while i<len(delta_list) and delta_list[i].startswith('+'):
                insertions += delta_list[i][1:]
                i += 1
            if insertions:
                res['Substitutions'].append((deletions,insertions))
            else:
                res['Deletions'].append(deletions)
        i += 1
    return res

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
diff = difflib.Differ()

hypothesis = []
with open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.eng", 'r') as infile:
    for i, line in enumerate(infile):
        hypothesis.append(tokenizer.decode(line.split()))
        if (i+1)%N == 0:
            aligned_hyp = mult_align(hypothesis)
            """for hyp in aligned_hyp:
                print("\t".join(w for w in hyp))"""
            for j in range(1, len(hypothesis)):
                print("\t".join(w for w in aligned_hyp[0]))
                print("\t".join(w for w in aligned_hyp[j]))
                #print("\t".join(w for w in list(diff.compare(hypothesis[0].split(), hypothesis[j].split()))))
                s = difflib.SequenceMatcher(None, hypothesis[0].split(), hypothesis[j].split())
                for tag, i1, i2, j1, j2 in s.get_opcodes():
                    if tag != 'equal':
                        print('{:7}  {!r:>8} --> {!r}'.format(tag, hypothesis[0].split()[i1:i2], hypothesis[j].split()[j1:j2]))
                #print(get_modifications(list(diff.compare(hypothesis[0].split(), hypothesis[j].split()))))
                print()
            print()
            hypothesis = []
