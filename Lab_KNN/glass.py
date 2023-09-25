import matplotlib.pyplot as plt

from numpy import sin, cos, exp, sqrt, pi
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def split_data(data, train_size):
    random_state = 2023
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    (x_train, x_test,
     y_train, y_test) = train_test_split(
            x, y,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
    )

    return x_train, x_test, y_train, y_test


def training(
        x_train, x_test,
        y_train, y_test,
        n_neighbors, weights
):
    model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm='kd_tree',
            leaf_size=200
    )
    model.fit(x_train, y_train)

    return model.score(x_test, y_test)


def plot(x_data, y_data, kernel_name):
    plt.plot(x_data, y_data)
    plt.ylabel('accuracy')
    plt.xlabel('number of neighbours')
    plt.title('Glass KNN with ' + kernel_name + ' kernel')
    plt.grid(True)
    plt.show()


class Kernels:
    # from https://en.wikipedia.org/wiki/Kernel_(statistics)
    def rectangular(self, distance):
        return 0.5

    def triangular(self, distance):
        return 1 - abs(distance)

    def epanechnikov(self, distance):
        return 0.75 * (1 - distance ** 2)

    def biweight(self, distance):
        return (15 / 16) * (1 - distance ** 2) ** 2

    def triweight(self, distance):
        return (35 / 32) * (1 - distance ** 2) ** 3

    def tricube(self, distance):
        return (70 / 81) * (1 - abs(distance) ** 3) ** 3

    def gaussian(self, distance):
        return (1 / sqrt(pi)) * exp(-0.5 * (distance ** 2))

    def cosine(self, distance):
        return (pi / 4) * cos(pi * distance / 2)

    def logistic(self, distance):
        return 1 / (exp(distance) + 2 + exp(-distance))

    def sigmoid(self, distance):
        return (2 / pi) / (exp(distance) + exp(-distance))

    def silverman(self, distance):
        sine = sin((abs(distance) / sqrt(2)) + (pi / 4))
        expon = exp(-abs(distance) / sqrt(2))
        return 0.5 * expon * sine


data_glass = read_csv('glass.xls', sep=',', header=0)

(x_train, x_test,
 y_train, y_test) = split_data(data_glass, train_size=0.6)

# number of neighbours from 1 to 13 without multiple of 7
neighbors_list = [
    1, 2, 3, 4, 5, 6,
    8, 9, 10, 11, 12, 13
]

functions_names = [
    'rectangular', 'triangular', 'epanechnikov',
    'biweight', 'triweight', 'tricube', 'gaussian',
    'cosine', 'logistic', 'sigmoid', 'silverman'
]

for kernel_name in functions_names:
    score_list = []
    kernel_func = getattr(Kernels(), kernel_name)

    for neighbors in neighbors_list:
        score = training(
                x_train, x_test, y_train, y_test,
                n_neighbors=neighbors, weights=kernel_func
        )
        score_list.append(score)

    plot(
            x_data=neighbors_list,
            y_data=score_list,
            kernel_name=kernel_name
    )
