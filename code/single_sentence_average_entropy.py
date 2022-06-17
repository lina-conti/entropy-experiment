from utils import *

#if __name__ == "__main__":
model = load_model("/home/lina/Desktop/Stage/transformer_wmt15_fr2en/transformer_wmt15_fr2en.yaml")

s = "▁l ' athlète ▁a ▁terminé ▁son ▁travail ▁."
t = "▁the ▁athlete ▁finished ▁his ▁work ▁."
src = encode_sentence(s.split(), model)

predicted_ids, df = decoding(model, src)

print("\npredicted sentence:")
print(to_tokens(predicted_ids, model))
print("\nreference sentence:\n" + t + "\n")

print("dataframe:")
print(df)

print("\naverage entropy of the decisions:")
print(np.mean(df["entropy"]))

print("\naverage difference between the highest and the second-highest probability:")
print(sum(difference_highest_second(prob_distribution) for prob_distribution in df["probas"]) / len(df["probas"]))

# distribution des entropies ?
df["entropy"].plot(kind="hist", xlabel="Entropy", ylabel="Frequency")
plt.show()

# vertical bar plot (entropy by decision)
df["entropy"].plot(kind="bar", xlabel="Number of the decision", ylabel="Entropy")
plt.show()
