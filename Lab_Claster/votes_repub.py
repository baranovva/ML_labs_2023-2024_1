import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

votes_repub = read_csv(filepath_or_buffer='votes_repub.csv', sep=',', header=0)

labels = votes_repub.iloc[:, 0]
votes_repub = votes_repub.iloc[:, 1:]

votes_repub_std = StandardScaler().fit_transform(votes_repub)

dist_matrix = linkage(votes_repub_std, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(Z=dist_matrix, labels=labels.values)
plt.ylabel('Height')
plt.show()
