"""
Compares the N best hypothesis predicted by the system for each sentence in the corpus:
deletions, insertions and substition necessary to go from one to the other
are written to a DataFrame
"""

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
from lingpy.align.multiple import mult_align
from sacremoses import MosesTokenizer
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
    mt = MosesTokenizer(lang='en')

    hypothesis = []
    s = 0
    with open(input_path, 'r') as infile:
        for l, line in enumerate(infile):
            hypothesis.append(line)
            if (l+1)%n == 0:
                s += 1
                for i in range(n-1):
                    for j in range(i, n):
                        if i == j:
                            continue
                        modifications = get_modifications(mt.tokenize(hypothesis[i], return_str=True).split(), hypothesis[j].split())
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
    mt = MosesTokenizer(lang='en')

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

    parser = argparse.ArgumentParser(description='Compares different translation \
    hypothesis and writes the results to a csv file.')
    parser.add_argument('n', help='number of translation hypothesis \
    for each sentence', type=int)
    parser.add_argument('corpus_path', help='path to the corpus to study')
    parser.add_argument('output_path', help='where to save the results')
    parser.add_argument('--compare_all', help='whether to compare all hypotheses \
    among themselves or only the 1-best to each of the others', action='store_true')
    args = parser.parse_args()

    file = f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}"
    if COMPARE_ALL:
        df = compare_all_hypotheses(file, N)
    else:
        df = compare_best_hypothesis(file, N)

    print(df)
    df.to_csv(f'/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/{CORPUS}.df.csv')
