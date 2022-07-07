import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

from sacrebleu.metrics import BLEU
import itertools

def hyps_to_lists(file_path: str, n: int) -> List[List[str]]:
    with open(file_path) as f:
        lines = f.readlines()

    corpus_hypotheses = []
    for i in range(n):
        corpus_hypotheses.append(list(itertools.islice(lines,i,None,n)))

    return corpus_hypotheses


if __name__ == "__main__":

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Started computing BLEU score.")

    parser = argparse.ArgumentParser(description='Computes the bleu score \
    using the translation hypotheses as references and the reference as hypothesis.')
    parser.add_argument("hypotheses_corpus", help="path to the corpus of tranlation hypotheses")
    parser.add_argument("reference_corpus", help="path to the reference corpus")
    parser.add_argument("n", help="number of translation hypotheses for each sentence", type=int)
    args = parser.parse_args()

    hyps = hyps_to_lists(args.hypotheses_corpus, args.n)

    with open(args.reference_corpus) as f:
        ref = f.readlines()

    bleu = BLEU()

    print(f"BLEU score when using the {args.hypotheses_corpus[args.hypotheses_corpus.rfind('/')+1:]}"
          f" corpus as references and the gold corpus as hypothesis: "
          f"{bleu.corpus_score(ref, hyps).score}.")

    datetime_obj = datetime.datetime.now()
    print(f"{datetime_obj.time()} - Finished computing BLEU score.")
