# Summer research internship 2022

These are the experiments I did during my internship with [Neuroviz](https://github.com/neuroviz/neuroviz), under the supervision of [Guillaume Wisniewski](https://github.com/guillaume-wisniewski).

I worked with machine translation models from French to English and from English to Portuguese, trained with the [JoeyNMT](https://github.com/joeynmt/joeynmt) framework. 

The experiments in `Entropy_experiments` aim to compare the entropy of the probability distributions when predicting the next token with greedy decoding or forced decoding.
Other scripts in this file measure the impact of mistakes in the target history on the system's performance.

The scripts in `Hesitation_experiments` exploit the n-best translation hypotheses to see where the system "hesitates" (between what words, constructions, etc).
It also contains code to get the n-best translation hypotheses with different decoding strategies (top-k sampling, greedy, beam search, ancestral sampling) and to measure the n-best hypotheses' diversity.

`Teacher_forcing_experiments` contains Jupyter notebooks for training machine translation models with or without teacher forcing and code to measure how well these models perform with mistakes in the target history.
