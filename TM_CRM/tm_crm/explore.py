from collections import Counter
import pandas as pd
import seaborn as sns

# Read data
df = pd.read_csv("dataset.tsv", delimiter='\t')

# Show summaries
print(df.head())
print(df.describe())

# Replace Korean with English & Encode
replace_dict = {
    "x4":{"저학력":"low", "중학력":"middle", "고학력":"high"},
    "x9":{"유선":"wired", '무선':"wireless"},
    "x11":{"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, \
           "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    }
df.replace(replace_dict, inplace=True)

# Incidences of each variables
for col in list(df):
    print()
    print(df[col].value_counts().sort_index())

# Function to plot each data
def plot_data(df, col, numeric_list):
    counts = df.groupby(df[col].name).size()
    counts_of_yes = df[df['y']=='yes'].groupby(df[col].name).size()
    yes_prob = counts_of_yes / counts

    s = pd.Series(Counter(df[col]))
    
    f, (ax1, ax2) = sns.plt.subplots(nrows=2)
    if col in numeric_list:
        sns.tsplot(data=s, ax=ax1)
        sns.tsplot(data=yes_prob, ax=ax2)
    else:
        sns.barplot(x=s.index.values, y=s, ax=ax1)
        sns.barplot(x=yes_prob.index.values, y=yes_prob, ax=ax2)
    ax1.set(ylabel="Incidences")
    ax2.set(ylabel="Probability of Yes")
    sns.plt.title(df[col].name)
    sns.plt.show()

# Plot incidences and the probability of yes for each variable
numeric_list = ['x1', 'x6']
for col in list(df):
    plot_data(df=df, col=col, numeric_list=numeric_list)
