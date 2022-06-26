"""
Compares the N best hypothesis predicted by the system for each sentence in the corpus:
deletions, insertions and substition necessary to go from one to the other
are written to a DataFrame
"""

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
from lingpy.align.multiple import mult_align
import sentencepiece as spm
import stanza
import difflib


def get_modifications(sentenceA, sentenceB):
    '''
    takes as input two lists of words and returns a dictionary
    with deletions, insertions and substition necessary for going from
    the first to the second
    '''
    res = {}
    matcher = difflib.SequenceMatcher(None, sentenceA, sentenceB)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            res[tag] = sentenceA[i1:i2], sentenceB[j1:j2]
    return res


def compare_all_hypotheses(input_path, n):
    """
    takes as input a file with a corpus of n-best translations
    outputs a dataframe with the modifications necessary to go from one
    hypothesis to the other (all the n-best hypotheses are compared two-by-two)
    """
    df = pd.DataFrame(columns=("sentence", "hypotheses", "replace", "insert", "delete"))
    tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')

    hypothesis = []
    s = 0
    with open(input_path, 'r') as infile:
        for l, line in enumerate(infile):
            hypothesis.append(tokenizer.decode(line.split()))
            if (l+1)%n == 0:
                s += 1
                for i in range(n-1):
                    for j in range(i, n):
                        if i == j:
                            continue
                        modifications = get_modifications(hypothesis[i].split(), hypothesis[j].split())
                        new_row = pd.DataFrame({"sentence": s,
                                    "hypotheses": [tuple([i+1,j+1])],
                                    "replace": [modifications['replace'] if 'replace' in modifications else None],
                                    "insert": [modifications['insert'][1] if 'insert' in modifications else None],
                                    "delete": [modifications['delete'][0] if 'delete' in modifications else None]
                                    })
                        df = pd.concat([df, new_row], ignore_index=True)
                hypothesis = []
    return df


def compare_best_hypothesis(input_path, n):
    """
    takes as input a file with a corpus of n-best translations
    outputs a dataframe with the modifications necessary to go from one
    hypothesis to the other (only the 1-best hypothesis is compared with all others)
    """
    df = pd.DataFrame(columns=("sentence", "hypotheses", "replace", "insert", "delete"))
    tokenizer = spm.SentencePieceProcessor(model_file='/home/lina/Desktop/Stage/tokenizers/en_tokenization.model')

    hypothesis = []
    s = 0
    with open(input_path, 'r') as infile:
        for l, line in enumerate(infile):
            hypothesis.append(tokenizer.decode(line.split()))
            if (l+1)%n == 0:
                s += 1
                for j in range(1, n):
                    modifications = get_modifications(hypothesis[0].split(), hypothesis[j].split())
                    new_row = pd.DataFrame({"sentence": s,
                                "hypotheses": [tuple([1,j+1])],
                                "replace": [modifications['replace'] if 'replace' in modifications else None],
                                "insert": [modifications['insert'][1] if 'insert' in modifications else None],
                                "delete": [modifications['delete'][0] if 'delete' in modifications else None]
                                })
                    df = pd.concat([df, new_row], ignore_index=True)
                hypothesis = []
    return df


if __name__ == "__main__":

    N = 5
    CORPUS = "newstest2014.pred"
    #POS_TAGS = False
    COMPARE_ALL = True

    file = f"/home/lina/Desktop/Stage/Modified_data/{CORPUS}.eng"
    if COMPARE_ALL:
        df = compare_all_hypotheses(file, N)
    else:
        df = compare_best_hypothesis(file, N)

    print(df)
    df.to_json(f'/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.df.json')
