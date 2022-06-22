'''
Finds the N ids with the highest probability, converts them to tokens and computes
- the proportion of tokens starting with _ (beginning of word tokens)
- the proportion of tokens starting with _ and preceded by a token starting in _
'''
N = 3
CORPUS = 'newstest2014'

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")

with Halo(text="Loading arrays", spinner="dots") as spinner:
    with open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/nbest_tokens.{CORPUS}.json", 'r') as infile:
        res_tokens = json.load(infile)
    """with open(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/nbest_lprobs.{CORPUS}.json", 'r') as infile:
        res_lprob = json.load(infile)"""
spinner.succeed()


print(f"number of decisions {len(res_tokens)}")

total_tokens = 0
bow_tokens = 0
bow_tokens_preceded = 0
preceded_by_bow = False
for top_ids in res_tokens:
    for i in range(N):
        if model.trg_vocab.itos[top_ids[i]].startswith('▁'):
            bow_tokens += 1
            if preceded_by_bow:
                bow_tokens_preceded += 1
        total_tokens += 1
    preceded_by_bow = model.trg_vocab.itos[top_ids[0]].startswith('▁')

print(f"\nnumber of tokens starting with \'_\': {bow_tokens}")
print(f"number of tokens starting with \'_\' and preceded by a token starting in \'_\': {bow_tokens_preceded}")
print(total_tokens)

print(f"\nLooking at the {N} tokens with highest probability for each decision when translating the {CORPUS} corpus")

print(f"proportion of tokens starting with \'_\': {round(bow_tokens*100/total_tokens, 2)}%")

print(f"proportion of tokens starting with \'_\' and preceded by a token starting in \'_\': {round(bow_tokens_preceded*100/total_tokens, 2)}%")
