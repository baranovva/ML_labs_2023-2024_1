import matplotlib.pyplot as plt

from numpy import sin, cos, exp, sqrt, pi
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def split_data(data, train_size, drop_attribute) -> object:
    random_state = 2023
    n_column = data.shape[1]

    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    if drop_attribute != None:
        x = x.drop(columns=drop_attribute)

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
        n_neighbors: int, weights,
        is_score: bool
):
    model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
    )
    model.fit(x_train, y_train)

    if is_score:
        return model.score(x_test, y_test)
    else:
        return model.predict(x_test)


def plot(x_data, y_data, kernel_name: str) -> None:
    plt.plot(x_data, y_data)
    plt.ylabel('accuracy')
    plt.xlabel('number of neighbours')
    plt.title('Glass KNN with ' + kernel_name + ' kernel')
    plt.grid(True)
    plt.show()


class Kernels:
    # kernels from https://en.wikipedia.org/wiki/Kernel_(statistics)
    def rectangular(self, distance):
        return 0.5

    def triangular(self, distance):
        return 1 - abs(distance)

    def epanechnikov(self, distance):
        return 0.75 * (1 - distance * distance)

    def biweight(self, distance):
        return (15 / 16) * (1 - distance * distance) ** 2

    def triweight(self, distance):
        return (35 / 32) * (1 - distance * distance) ** 3

    def tricube(self, distance):
        return (70 / 81) * (1 - abs(distance) ** 3) ** 3

    def gaussian(self, distance):
        return (1 / sqrt(pi)) * exp(-0.5 * (distance * distance))

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

print(*data_glass.keys())

(x_train, x_test,
 y_train, y_test) = split_data(
        data_glass,
        train_size=0.6,
        drop_attribute=None
)

# task 1
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
                x_train, x_test,
                y_train, y_test,
                n_neighbors=neighbors,
                weights=kernel_func,
                is_score=True
        )
        score_list.append(score)

    plot(
            x_data=neighbors_list,
            y_data=score_list,
            kernel_name=kernel_name
    )

# task 2
sample_glass = {'RI': [1.515],
                'Na': [11.7],
                'Mg': [1.01],
                'Al': [1.19],
                'Si': [72.59],
                'K': [0.43],
                'Ca': [11.44],
                'Ba': [0.02],
                'Fe': [0.1]}
sample_glass = DataFrame(data=sample_glass)

predict = training(
        x_train=x_train, x_test=sample_glass,
        y_train=y_train, y_test=None,
        n_neighbors=9, weights='distance',
        is_score=False
)

print(*predict)

# task 3
score_list = []
x_test_keys = list(x_test.keys())

for dropped_attribute in x_test_keys:
    (x_train, x_test,
     y_train, y_test) = split_data(
            data_glass, train_size=0.8,
            drop_attribute=dropped_attribute
    )

    score = training(
            x_train=x_train, x_test=x_test,
            y_train=y_train, y_test=y_test,
            n_neighbors=11, weights='distance',
            is_score=True
    )

    score_list.append(score)

plt.plot(x_test_keys, score_list)
plt.xlabel('dropped attribute')
plt.ylabel('accuracy')
plt.title('Glass KNN drop test')
plt.grid(True)
plt.show()
