"""
Compares the N best hypothesis predicted by the system for each sentence in the corpus:
deletions, insertions and substition necessary to go from one to the other
are written to a DataFrame
"""

import sys
sys.path.insert(0, '/home/lina/Desktop/Stage/Experiences/code')
from utils import *

with Halo(text="Loading dataframe", spinner="dots") as spinner:
    df = pd.read_json(f"/home/lina/Desktop/Stage/Experiences/results/Hesitation_experiments/newstest2014.pred.df.json")


print(f"Number of hesitations between two possible formulations: {df['replace'].count()}")
print(f"Of these, \n"
    f"- {pd.Series((len(l[0])==1) ^ (len(l[1])==1) for l in df['replace'] if l).sum() * 100 / df['replace'].count():.2f}"
    f"% are hesitations between one word and a multiword expression \n"
    f"- {pd.Series((len(l[0])==1) and (len(l[1])==1) for l in df['replace'] if l).sum() * 100 / df['replace'].count():.2f}"
    f"% are hesitations between two single words \n"
    f"- {pd.Series((len(l[0]) > 1) and (len(l[1]) > 1) for l in df['replace'] if l).sum() * 100 / df['replace'].count():.2f}"
    f"% are hesitations between two multiword expressions")
print()
print(f"Number of hesitations between including a word or expression or not:"
    f"{df['insert'].count() + df['delete'].count()}")
print(f"Of these, \n"
    f"- {(pd.Series(len(l)==1 for l in df['delete'] if l).sum() + pd.Series(len(l)==1 for l in df['insert'] if l).sum()) * 100 / (df['delete'].count() + df['insert'].count()):.2f}"
    f"% are for single words \n"
    f"- {(pd.Series(len(l)>1 for l in df['delete'] if l).sum() + pd.Series(len(l)>1 for l in df['insert'] if l).sum()) * 100 / (df['delete'].count() + df['insert'].count()):.2f}"
    f"% are for multiword expressions \n")
