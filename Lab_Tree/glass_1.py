import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

random_state = 2023


def split_data(data, train_size=0.6, random_state=2023) -> object:
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


data_glass = read_csv(filepath_or_buffer='Glass.csv', sep=',', header=0)

x_train, x_test, y_train, y_test = split_data(data_glass, train_size=0.9)

tree = DecisionTreeClassifier(random_state=random_state).fit(x_train, y_train)
print(tree.score(x_train, y_train), tree.score(x_test, y_test))

plt.figure(figsize=(10, 10))
plot_tree(tree, filled=True, fontsize=5, rounded=True, impurity=False)
plt.show()

param_grid = {
    'criterion': ('gini', 'entropy', 'log_loss'),
    'max_depth': np.arange(2, 6),
    'min_samples_split': np.arange(2, 7),
    'min_samples_leaf': np.arange(1, 6)
}
search = GridSearchCV(tree, param_grid=param_grid, n_jobs=-8).fit(x_train, y_train)
best_parameters = search.best_params_
print("The best:", best_parameters, 'with score:', search.best_score_)

tree_my = DecisionTreeClassifier(
        random_state=random_state,
        criterion=best_parameters['criterion'],
        max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf']
).fit(x_train, y_train)

print(tree_my.score(x_train, y_train), tree_my.score(x_test, y_test))

plt.figure(figsize=(10, 10))
plot_tree(tree_my, filled=True, fontsize=5, rounded=True, impurity=False)
plt.show()
