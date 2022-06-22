"""
old version that doesn't analyse forced and normal decoding side by side
"""

from utils import *

with Halo(text="Loading dataframe", spinner="dots") as spinner:
    df = pd.read_pickle('/home/lina/Desktop/Stage/Experience_entropie/results/entropies_newstest2014.pickle')

print(df.shape)

df['token_position'] = df['token_position'].astype(int)

# entropies moyennes
print("average entropy of the decisions:")
print(np.mean(df["entropy"]))

# différence moyenne entre les deux plus fortes probas (probability distributions need to be saved)
# print("\naverage difference between the highest and the second-highest probability:")
# print(log(np.mean([difference_highest_second(prob_distribution) for prob_distribution in df["log_probas"]])))


# distribution des entropies
df["entropy"].plot(kind="hist", title="Decision entropy when translating newstest2014 corpus", bins=50, color='purple')
plt.xlabel("Entropy")
plt.ylabel("Frequency")
#plt.savefig("/home/lina/Desktop/Stage/Experience_entropie/results/ent_dist_newstest2014_forced.png")
plt.show()

"""
# vertical bar plot (entropy by decision)
df.groupby(by="token_position").mean()["entropy"].plot(kind="bar", xlabel="Token position", ylabel="Entropy", title="Average entropy by token position for the newstest2014 corpus translated with forced decoding", color='purple')
#plt.savefig("/home/lina/Desktop/Stage/Experience_entropie/results/ent_by_decision_newstest2014.png")
plt.show()
"""

ax = sns.barplot(x="token_position", y="entropy", data=df, color='orchid',)
ax.set(xlabel="Token position", ylabel="Entropy", title="Average entropy by token position for the newstest2014 corpus translated")
plt.show()

# entropy by part of sentence
df = df.assign(sentence_part = lambda x: x.token_position / x.sentence_length * 100)
df['sentence_part'] = pd.cut(df['sentence_part'], 10, labels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
"""
df.groupby(by="sentence_part")["entropy"].mean().plot(kind="bar", xlabel="Token position normalized by sentence length", ylabel="Entropy", color='purple', title="Average entropy by part of the sentence for the newstest2014 corpus translated with forced decoding")
#plt.savefig("/home/lina/Desktop/Stage/Experience_entropie/results/ent_by_decision_normalized_newstest2014_forced.png")
plt.show()
"""
ax = sns.barplot(x="sentence_part", y="entropy", data=df, color='orchid',)
ax.set(xlabel="Token position normalized by sentence length", ylabel="Entropy", title="Average entropy by part of the sentence for the newstest2014 corpus translated")
plt.show()
