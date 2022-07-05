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
import stanza


def get_modifications(sentenceA, sentenceB, nlp):
    '''
    takes as input two lists of words and a stanza Pipeline
    with POS tagger and returns two dictionaries with deletions, insertions
    and substition necessary for going from the first list to the second.
    the first dictionary contains the editions in words and the second
    the editions in part of speech tags.
    '''
    docA = nlp(sentenceA)
    docB = nlp(sentenceB)
    posA = [word.upos for word in docA.sentences[0].words]
    posB = [word.upos for word in docB.sentences[0].words]
    wordsA = [word.text for word in docA.sentences[0].words]
    wordsB = [word.text for word in docB.sentences[0].words]
    matcher = difflib.SequenceMatcher(None, wordsA, wordsB)
    word_editions = defaultdict(lambda: [])
    pos_editions = defaultdict(lambda: [])
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            word_editions[tag].append(tuple([" ".join(wordsA[i1:i2]), " ".join(wordsB[j1:j2])]))
            pos_editions[tag].append(tuple([" ".join(posA[i1:i2]), " ".join(posB[j1:j2])]))
        if tag == 'delete':
            word_editions[tag].append(" ".join(wordsA[i1:i2]))
            pos_editions[tag].append(" ".join(posA[i1:i2]))
        if tag == 'insert':
            word_editions[tag].append(" ".join(wordsB[j1:j2]))
            pos_editions[tag].append(" ".join(posB[j1:j2]))
    similarity = matcher.ratio()
    return word_editions, pos_editions, similarity


def compare_all_hypotheses(input_path, n):
    """
    takes as input a file with a corpus of n-best translations
    outputs a dataframe with the modifications necessary to go from one
    hypothesis to the other (all the n-best hypotheses are compared two-by-two)
    """
    df = pd.DataFrame()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')

    hypothesis = []
    s = 0
    with open(input_path, 'r') as infile:
        for l, line in enumerate(infile):
            hypothesis.append(line)
            if (l+1)%n == 0:
                s += 1
                with Halo(text=f"Comparing hypotheses for sentence {s}.", spinner="dots"):
                    for i in range(n-1):
                        for j in range(i, n):
                            if i == j:
                                continue
                            modifications, modifications_pos, similarity = get_modifications(hypothesis[i], hypothesis[j], nlp)
                            new_row = pd.DataFrame({"sentence": s,
                                        "hypotheses": [tuple([i+1,j+1])],
                                        "sequence_similarity": similarity,
                                        "replace": [modifications['replace'] if 'replace' in modifications else None],
                                        "insert": [modifications['insert'] if 'insert' in modifications else None],
                                        "delete": [modifications['delete'] if 'delete' in modifications else None],
                                        "replace_pos": [modifications_pos['replace'] if 'replace' in modifications_pos else None],
                                        "insert_pos": [modifications_pos['insert'] if 'insert' in modifications_pos else None],
                                        "delete_pos": [modifications_pos['delete'] if 'delete' in modifications_pos else None]
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
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')

    hypothesis = []
    s = 0
    with open(input_path, 'r') as infile:
        for l, line in enumerate(infile):
            hypothesis.append(line)
            if (l+1)%n == 0:
                s += 1
                with Halo(text=f"Comparing hypotheses for sentence {s}.", spinner="dots"):
                    for j in range(1, n):
                        modifications, modifications_pos, similarity = get_modifications(hypothesis[0], hypothesis[j], nlp)
                        new_row = pd.DataFrame({"sentence": s,
                                    "hypotheses": [tuple([1,j+1])],
                                    "sequence_similarity": similarity,
                                    "replace": [modifications['replace'] if 'replace' in modifications else None],
                                    "insert": [modifications['insert'] if 'insert' in modifications else None],
                                    "delete": [modifications['delete'] if 'delete' in modifications else None],
                                    "replace_pos": [modifications_pos['replace'] if 'replace' in modifications_pos else None],
                                    "insert_pos": [modifications_pos['insert'] if 'insert' in modifications_pos else None],
                                    "delete_pos": [modifications_pos['delete'] if 'delete' in modifications_pos else None]
                                    })
                        df = pd.concat([df, new_row], ignore_index=True)
                hypothesis = []
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compares different translation \
    hypothesis and writes the results to a file.')
    parser.add_argument('n', help='number of translation hypothesis \
    for each sentence', type=int)
    parser.add_argument('corpus_path', help='path to the corpus to study')
    parser.add_argument('output_path', help='where to save the results')
    parser.add_argument('--compare_all', help='whether to compare all hypotheses \
    among themselves or only the 1-best to each of the others', action='store_true')
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started comparing translation hypotheses.")

    if args.compare_all:
        df = compare_all_hypotheses(args.corpus_path, args.n)
    else:
        df = compare_best_hypothesis(args.corpus_path, args.n)

    if args.output_path.endswith("csv"):
        df.to_csv(args.output_path)
    elif args.output_path.endswith("json"):
        df.to_json(args.output_path)
    else:
        print("Extension not recognized, results could not be saved.")
        exit()

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished comparing translation hypotheses.\n"
          f"Results saved to {args.output_path}")
