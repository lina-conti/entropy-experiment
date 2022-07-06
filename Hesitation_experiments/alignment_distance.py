import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *
import simalign
from scipy.stats import kendalltau

def alignment_distance(sentenceA, sentenceB, aligner):
    '''
    takes as input two lists of words and a simalign SentenceAligner
    and returns the alignment distance (kendalltau of the alignements)
    '''
    alignements = aligner.get_word_aligns(sentenceA, sentenceB)

    # inter (argmax) works better for close language pairs according to the article
    # so should work better for the same language, it tends to give higher kendall
    # tau values wrt other alignement methods
    l1 = [x1 for x1,x2 in alignements["inter"]]
    l2 = [x2 for x1,x2 in alignements["inter"]]

    correlation, p_value = kendalltau(l1, l2)
    return correlation


def average_alignement_distances(input_path, n, compare_all):
    """
    takes as input a file with a corpus of n-best translations
    outputs the average alignement distance between hypotheses
    and the average minimum and maximum alignment distance by source sentence
    """
    df = pd.DataFrame()
    aligner = simalign.SentenceAligner()
    average_dist = 0
    average_min = 0
    average_max = 0
    nb_comparisons = 0

    hypotheses = []
    s = 0
    with open(input_path, 'r') as infile:
        for l, line in enumerate(infile):
            hypotheses.append(line)
            if (l+1)%n == 0:
                s += 1
                min_dist = 1.
                max_dist = -1.
                with Halo(text=f"Comparing hypotheses for sentence {s}.", spinner="dots"):
                    for i in range(n-1):
                        if compare_all:
                            for j in range(i, n):
                                if i == j:
                                    continue
                                align_dist = alignment_distance(hypotheses[i], hypotheses[j], aligner)
                                if not pd.isna(align_dist):
                                    average_dist += align_dist
                                    if align_dist > max_dist:
                                        max_dist = align_dist
                                    if align_dist < min_dist:
                                        min_dist = align_dist
                                    nb_comparisons += 1
                        else:
                            align_dist = alignment_distance(hypotheses[0], hypotheses[i+1], aligner)
                            if not pd.isna(align_dist):
                                average_dist += align_dist
                                if align_dist > max_dist:
                                    max_dist = align_dist
                                if align_dist < min_dist:
                                    min_dist = align_dist
                                nb_comparisons += 1
                    average_min += min_dist
                    average_max += max_dist
                    hypotheses = []
    average_dist = average_dist / nb_comparisons
    average_min = average_min / s
    average_max = average_max / s
    return average_dist, average_min, average_max


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculates the alignment \
    distance between different translation hypotheses and outputs the average \
    alignement distance between hypotheses and the average minimum and maximum \
    alignment distance by source sentence.')
    parser.add_argument('n', help='number of translation hypotheses \
    for each sentence', type=int)
    parser.add_argument('corpus_path', help='path of the corpus to be used')
    parser.add_argument('--compare_all', help='whether to compare all hypotheses \
    among themselves or only the 1-best to each of the others', action='store_true')
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started comparing translation hypotheses.")

    average_dist, average_min, average_max = average_alignement_distances(
                                    args.corpus_path, args.n, args.compare_all)

    print(f"Results for {args.corpus_path[args.corpus_path.rfind('/')+1:]} "
          f"corpus with compare_all = {args.compare_all}")
    print(f"Average alignment distance: {average_dist}")
    print(f"Average maximum alignment distance: {average_max}")
    print(f"Average minimum alignment distance: {average_min}")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished comparing translation hypotheses.")
