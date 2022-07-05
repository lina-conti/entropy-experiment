"""
Analyses the dataframes produced by corpus_average_entropy.py and
corpus_entropy_forced.py. For forced and normal decoding:
- plots the distributions of the entropies;
- plots the average entropy by position (normalized by sentence length or not).
"""

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("corpus", help="name of the corpus to use")
args = parser.parse_args()

with Halo(text="Loading dataframe", spinner="dots") as spinner:
    df = pd.read_json(f"/home/lina/Desktop/Stage/Experiences/results/Entropy_experiments/dataframes/entropies_{args.corpus}.json")
with Halo(text="Loading dataframe", spinner="dres_tokensots") as spinner:
    df_forced = pd.read_json(f"/home/lina/Desktop/Stage/Experiences/results/Entropy_experiments/dataframes/entropies_{args.corpus}_forced.json")

# entropies moyennes
print("average entropy of the decisions:")
print(df["entropy"].mean())
# entropies moyennes
print("average entropy of the decisions forced:")
print(df_forced["entropy"].mean())

print(df_forced['token_position'].argmax())
print(df['token_position'].argmax())

# distribution des entropies décodage normal
df["entropy"].plot(kind="hist", title=f"Decision entropy when translating {args.corpus} corpus with greedy decoding", bins=50, color='blue')
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.show()

# distribution des entropies décodage forcé
df_forced["entropy"].plot(kind="hist", title=f"Decision entropy when translating {args.corpus} corpus with forced decoding", bins=50, color='orange')
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.show()

# merging the two dfs
df = df.assign(decoding_mode = 'normal')
df_forced = df_forced.assign(decoding_mode = 'forced')
df = pd.concat([df, df_forced])
df['token_position'] = df['token_position'].astype(int)
df['sentence_length'] = df['sentence_length'].astype(int)

# entropy by position
ax = sns.barplot(x="token_position", y="entropy", hue="decoding_mode", data=df)
ax.set(xlabel="Token position", ylabel="Entropy", title=f"Average entropy by token position for the {args.corpus} corpus")
plt.show()

# entropy by part of sentence
df = df.assign(sentence_part = lambda x: x.token_position / x.sentence_length * 100)
df['sentence_part'] = pd.cut(df['sentence_part'], 10, labels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax = sns.barplot(x="sentence_part", y="entropy", data=df, hue="decoding_mode")
ax.set(xlabel="Token position normalized by sentence length", ylabel="Entropy", title=f"Average entropy by part of the sentence for the {args.corpus} corpus")
plt.show()
