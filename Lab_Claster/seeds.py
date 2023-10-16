from pandas import read_csv
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import davies_bouldin_score, silhouette_score

seeds = read_csv(filepath_or_buffer='seeds_dataset.txt', sep='\t', header=None)

labels = seeds.iloc[:, 7].values
seeds = seeds.iloc[:, :7].values

random_state = 2023

print('KMeans')
model = KMeans(n_clusters=3, n_init='auto', random_state=random_state).fit(seeds)
labels = model.labels_
print(davies_bouldin_score(seeds, labels=labels), silhouette_score(seeds, labels=labels, random_state=random_state))

print('AgglomerativeClustering')
model = AgglomerativeClustering(n_clusters=3, metric='euclidean').fit(seeds)
labels = model.labels_
print(davies_bouldin_score(seeds, labels=labels), silhouette_score(seeds, labels=labels, random_state=random_state))

print('Birch')
model = Birch(n_clusters=3).fit(seeds)
labels = model.labels_
print(davies_bouldin_score(seeds, labels=labels), silhouette_score(seeds, labels=labels, random_state=random_state))
