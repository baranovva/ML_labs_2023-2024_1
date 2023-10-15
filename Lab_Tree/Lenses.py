from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def split_data(data, train_size=0.9, random_state=2023) -> object:
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


data = read_csv(filepath_or_buffer='Lenses.txt', sep='  ', header=None)

x_train, x_test, y_train, y_test = split_data(data, train_size=0.95)

random_state = 2023
tree = DecisionTreeClassifier(random_state=random_state).fit(x_train, y_train)
print(tree.score(x_train, y_train), tree.score(x_test, y_test))

plt.figure(figsize=(10, 10))
plot_tree(tree, filled=True, fontsize=15, rounded=True, impurity=False)
plt.show()

data_sample = DataFrame(data={'0': [2], '1': [1], '2': [2], '3': [1]})
print(*tree.predict(data_sample))
