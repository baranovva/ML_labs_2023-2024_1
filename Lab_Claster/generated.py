import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score

random_state = 2023

data, labels = make_blobs(n_samples=1200, centers=3, cluster_std=[[4, 2], [2, 7], [4, 1]], random_state=random_state)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()

data_std = StandardScaler().fit_transform(data)

for metric in ['euclidean', 'manhattan', 'l1', 'l2', 'cosine']:
    for case in (data, data_std):
        model = AgglomerativeClustering(n_clusters=3, linkage="average", metric=metric).fit(case)
        print(metric)
        print(davies_bouldin_score(case, labels=model.labels_),
              silhouette_score(case, labels=model.labels_, random_state=random_state))
