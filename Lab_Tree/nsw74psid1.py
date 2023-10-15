import numpy as np
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


def split_data(data, train_size=0.9, random_state=2023) -> object:
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


data = read_csv(filepath_or_buffer='nsw74psid1.csv', sep=',', header=0)

x_train, x_test, y_train, y_test = split_data(data)

# tree
param_grid = {
    'criterion': ('squared_error', 'friedman_mse', 'absolute_error', 'poisson'),
    'max_depth': np.arange(2, 8),
    'min_samples_split': np.arange(2, 8),
    'min_samples_leaf': np.arange(1, 9)
}

random_state = 2023
tree = DecisionTreeRegressor(random_state=random_state)

search = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1).fit(x_train, y_train)
best_parameters = search.best_params_
print("The best:", best_parameters, 'with score:', search.best_score_)

tree_my = DecisionTreeRegressor(
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

# SVM
param_grid = {
    'C': np.arange(-5, 6),
    'kernel': ("poly", "rbf", "sigmoid"),
    'degree': np.arange(1, 8)
}

search = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1).fit(x_train, y_train)
best_parameters = search.best_params_
print("The best:", best_parameters, 'with score:', search.best_score_)

svm_my = SVR(
        C=best_parameters['C'],
        kernel=best_parameters['kernel'],
        degree=best_parameters['degree']
).fit(x_train, y_train)

print(svm_my.score(x_train, y_train), svm_my.score(x_test, y_test))

# regression
linear_my = LinearRegression(n_jobs=-1).fit(x_train, y_train)

print(linear_my.score(x_train, y_train), linear_my.score(x_test, y_test))
