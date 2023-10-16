import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

animals = read_csv(filepath_or_buffer='animals.csv', sep=',', header=0)

labels = animals.iloc[:, 0]
animals = animals.iloc[:, 1:]

animals_std = StandardScaler().fit_transform(animals)

dist_matrix = linkage(animals_std, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(Z=dist_matrix, labels=labels.values)
plt.ylabel('Height')
plt.show()
