"""
Compares the N best hypothesis predicted by the system:
- hypothesis are detokenized, aligned and written to a file
- hypothesis can also be annotated for POS (optional)
"""

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
from lingpy.align.multiple import mult_align
import sentencepiece as spm
import stanza
import difflib

def align_hypothesis(input_path, output_path, n, POS_tags=False):
    '''
    detokenizes, aligns and writes hypothesis to a file,
    hypothesis can also be annotated for POS (optional)
    '''
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
    tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
    hypothesis = []
    with open(input_path, 'r') as infile:
        with open(output_path, 'w') as outfile:
            for i, line in enumerate(infile):
                hypothesis.append(tokenizer.decode(line.split()))
                if (i+1)%n == 0:
                    aligned_hyp = mult_align(hypothesis)
                    for j in range(len(hypothesis)):
                        outfile.write(" \t".join(word.ljust(10) for word in aligned_hyp[j]))
                        outfile.write("\n")
                        if POS_TAGS:
                            doc = nlp(hypothesis[j])
                            outfile.write(" \t".join(word.upos.ljust(10) for sent in doc.sentences for word in sent.words))
                            outfile.write("\n")
                    hypothesis = []
                    outfile.write("\n")


def compare_hypothesis(input_path, output_path, n):
    '''
    compares each of the n-best hypothesis in the input_path file with the
    1-best hypothesis using difflib, results are written to the output_path file
    '''
    diff = difflib.Differ()
    tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')
    hypothesis = []
    with open(input_path, 'r') as infile:
        with open(output_path, 'w') as outfile:
            for i, line in enumerate(infile):
                hypothesis.append(line)
                if line.strip() == "":
                    for j in range(1, len(hypothesis)-1):
                        res = list(diff.compare(hypothesis[0].split(), hypothesis[j].split()))
                        outfile.write(" ".join(word for word in res))
                        outfile.write("\n")
                    hypothesis = []
                    outfile.write("\n")

def get_modifications(delta_list):
    '''
    takes as input a list obtained with Differ.compare and returns a dictionary
    with deleted words, inserted words and substituted words
    '''
    res = {'Substitutions': [], 'Insertions': [], 'Deletions': []}
    for i in range(len(delta_list)):
        if delta_list[i].startswith('+'):
            res['Substitutions'].append(delta_list[i][2:])
        elif delta_list[i].startswith('-'):
            if True:
                pass


if __name__ == "__main__":

    N = 5
    CORPUS = "X-a-fini.small.pred"
    POS_TAGS = False

    align_hypothesis(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.eng",\
     f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.aligned.eng",\
     N, POS_tags=POS_TAGS)

    compare_hypothesis(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.aligned.eng",\
     f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.comp.eng", N)
