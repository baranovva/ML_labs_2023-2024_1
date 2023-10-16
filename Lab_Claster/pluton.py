from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

data = read_csv(filepath_or_buffer='pluton.csv', sep=',', header=0)

random_state = 2023
for max_iter in range(1, 4):
    model = KMeans(n_clusters=3, init='random', n_init='auto', random_state=random_state, max_iter=max_iter).fit(data)

    labels = model.labels_
    print(davies_bouldin_score(data, labels=labels), silhouette_score(data, labels=labels, random_state=random_state))
