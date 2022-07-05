import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Started translating the corpus.")

parser = argparse.ArgumentParser(description='Compares the mistakes \
when using forced decoding versus greedy decoding.')
parser.add_argument("source_corpus", help="path to the source corpus")
parser.add_argument("target_corpus", help="path to the target corpus")
args = parser.parse_args()

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")
max_output_length = load_config("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")["training"]["max_output_length"]


df, correct_gold, correct_predicted = mistake_stats(args.source_corpus,
                        args.target_corpus, model, max_output_length)

print(df)
print(f"\nNumber of tokens that are correctly predicted with forced decoding but"
      f" not with greedy decoding: {correct_gold}.")
print(f"\nNumber of tokens that are correctly predicted with greedy decoding but"
      f" not with forced decoding: {correct_gold}.\n")

datetime_obj = datetime.datetime.now()
print(f"{datetime_obj.time()} - Finished analysing the translations.")
