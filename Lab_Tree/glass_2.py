import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV


def split_data(data, train_size=0.9, random_state=2023) -> object:
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


data_glass = read_csv(filepath_or_buffer='Glass.csv', sep=',', header=0)

x_train, x_test, y_train, y_test = split_data(data_glass)

param_grid = {
    'criterion': ('gini', 'entropy', 'log_loss'),
    'max_depth': np.arange(2, 6),
    'min_samples_split': np.arange(2, 7),
    'min_samples_leaf': np.arange(1, 6)
}

random_state = 2023
tree = DecisionTreeClassifier(random_state=random_state)

search = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1).fit(x_train, y_train)
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

sample_glass = DataFrame(data={'RI': [1.515],
                               'Na': [11.7],
                               'Mg': [1.01],
                               'Al': [1.19],
                               'Si': [72.59],
                               'K': [0.43],
                               'Ca': [11.44],
                               'Ba': [0.02],
                               'Fe': [0.1]})

print(*tree_my.predict(sample_glass))

plt.figure(figsize=(10, 10))
plot_tree(tree_my, filled=True, fontsize=5, rounded=True, impurity=False)
plt.show()
