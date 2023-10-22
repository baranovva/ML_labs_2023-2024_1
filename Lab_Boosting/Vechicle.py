import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from pandas import read_csv
from sklearn.model_selection import train_test_split


def split_data(data, train_size=0.7, random_state=2023) -> object:
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


data_vechicle = read_csv(filepath_or_buffer='../Lab_Boosting/vehicle.csv', sep=',', header=0)
x_train, x_test, y_train, y_test = split_data(data_vechicle)

n_trees = range(1, 302, 10)
test_error = []
for n in n_trees:
    model = AdaBoostClassifier(n_estimators=n).fit(x_train, y_train)
    test_error.append(1 - model.score(x_test, y_test))

plt.plot(n_trees, test_error)
plt.xlabel('Number of trees')
plt.ylabel('Test error')
plt.grid(True)
plt.show()
