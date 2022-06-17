from utils import *

with Halo(text="Loading dataframe", spinner="dots") as spinner:
    df = pd.read_pickle('/home/lina/Desktop/Stage/Experience_entropie/results/dataframes/entropies_newstest2014.pickle')
with Halo(text="Loading dataframe", spinner="dots") as spinner:
    df_forced = pd.read_pickle('/home/lina/Desktop/Stage/Experience_entropie/results/dataframes/entropies_newstest2014_forced.pickle')


# entropies moyennes
print("average entropy of the decisions:")
print(np.mean(df["entropy"]))
# entropies moyennes
print("average entropy of the decisions forced:")
print(np.mean(df_forced["entropy"]))

# merging the two dfs
df = df.assign(decoding_mode = 'normal')
df_forced = df_forced.assign(decoding_mode = 'forced')
df = pd.concat([df, df_forced])
df['token_position'] = df['token_position'].astype(int)
df['sentence_length'] = df['sentence_length'].astype(int)

"""
print(df[df['token_position'] > 40].count())
print(df.count())
print((df[df['token_position'] > 40].count() / df.count()) * 100)
"""
# distribution des entropies
df["entropy"].plot(kind="hist", title="Decision entropy when translating newstest2014 corpus", bins=50, color='purple')
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.show()

# entropy by position
ax = sns.barplot(x="token_position", y="entropy", hue="decoding_mode", data=df[df['token_position'] <= 71])
ax.set(xlabel="Token position", ylabel="Entropy", title="Average entropy by token position for the newstest2014 corpus")
plt.show()

# entropy by part of sentence
df = df.assign(sentence_part = lambda x: x.token_position / x.sentence_length * 100)
df['sentence_part'] = pd.cut(df['sentence_part'], 10, labels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax = sns.barplot(x="sentence_part", y="entropy", data=df, hue="decoding_mode")
ax.set(xlabel="Token position normalized by sentence length", ylabel="Entropy", title="Average entropy by part of the sentence for the newstest2014 corpus")
plt.show()
