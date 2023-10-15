import numpy as np
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV


def data_split(data_train: object, data_test: object) -> object:
    n_column = data_train.shape[1]

    x_train = data_train.iloc[:, 0:n_column - 1]
    y_train = data_train.iloc[:, -1]

    x_test = data_test.iloc[:, 0:n_column - 1]
    y_test = data_test.iloc[:, -1]

    return x_train, y_train, x_test, y_test


data_train = read_csv(filepath_or_buffer='svmdata4.txt', sep='	', header=0)
data_test = read_csv(filepath_or_buffer='svmdata4test.txt', sep='	', header=0)

x_train, y_train, x_test, y_test = data_split(data_train, data_test)

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

plt.figure(figsize=(10, 10))
plot_tree(tree_my, filled=True, fontsize=13, rounded=True, impurity=False)
plt.show()
