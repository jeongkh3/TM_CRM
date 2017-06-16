import pandas as pd
from copy import deepcopy
from kmodes import kprototypes
from sklearn.preprocessing import MinMaxScaler
from util import modify_max_outlier
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 99

# Read data
df = pd.read_csv("dataset.tsv", delimiter='\t')

replace_dict = {
    "x4":{"저학력":"low", "중학력":"middle", "고학력":"high"}
    }

df.replace(replace_dict, inplace=True)

df.drop(['x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'y'], axis=1, inplace=True)

df[['x1', 'x6']] = df[['x1', 'x6']].astype(float)


# Modify Out-lier
#plotOutlier(df['x6'].sample(1000))
df.x6 = modify_max_outlier(df, col='x6', max=6000)

raw_df = deepcopy(df)

mms = MinMaxScaler()
df[['x1', 'x6']] = mms.fit_transform(df[['x1', 'x6']])

print(df.head())

# Sampling
X_sample = df.sample(n=4000, replace=False, random_state=RANDOM_SEED)

# Run algorithm with the different number of clusters
k_range = range(2, 16, 2)
cost = []
for k in k_range:
    kproto = kprototypes.KPrototypes(n_clusters=k, init='Cao')
    clusters = kproto.fit_predict(X_sample.as_matrix(), categorical=[1, 2, 3, 4, 6, 7])
    cost.append(kproto.cost_)

# Plot cost with the number of clusters
sns.set_style('whitegrid')
plt.plot(k_range, cost)
plt.xlabel('k: # of clusters')
plt.ylabel('cost')
sns.plt.show()

# Run algorithm with whole data
kp = kprototypes.KPrototypes(n_clusters=4, init='Cao')
kp_clusters = kp.fit_predict(df.as_matrix(), categorical=[1, 2, 3, 4, 6, 7])

# Print training statistics
print(kp.cost_)
# Print cluster centroids of the trained model.
print(kp.cluster_centroids_)



# Add a column for convenience
raw_df['cluster'] = kp_clusters

for i in sorted(list(set(kp_clusters))):
    print('\n Cluster: %i'%(i))
    print(raw_df[raw_df['cluster']==i].describe())


# Function to make table with variables and clusters
def make_cluster_df(data, col, clusters, cluster_col_name='cluster'):
    data_list = []
    for i in sorted(list(set(clusters))):
        data_list.append(data[data[cluster_col_name]==i].groupby(col).size().values.tolist())
    
    new_df = pd.DataFrame(data_list)
    new_df.columns = data[col].unique().tolist()
    return new_df


# Show incidences of each category and cluster
x2_df = make_cluster_df(data=raw_df, col='x2', clusters=kp_clusters, cluster_col_name='cluster')
print(x2_df)

x3_df = make_cluster_df(data=raw_df, col='x3', clusters=kp_clusters, cluster_col_name='cluster')
print(x3_df)

x4_df = make_cluster_df(data=raw_df, col='x4', clusters=kp_clusters, cluster_col_name='cluster')
print(x4_df)

x5_df = make_cluster_df(data=raw_df, col='x5', clusters=kp_clusters, cluster_col_name='cluster')
print(x5_df)

x7_df = make_cluster_df(data=raw_df, col='x7', clusters=kp_clusters, cluster_col_name='cluster')
print(x7_df)

x8_df = make_cluster_df(data=raw_df, col='x8', clusters=kp_clusters, cluster_col_name='cluster')
print(x8_df)
