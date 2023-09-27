import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


def data_split(data_train, data_test) -> object:
    n_column = data_train.shape[1]

    x_train = data_train.iloc[:, 0:n_column - 1]
    y_train = data_train.iloc[:, -1]

    x_test = data_test.iloc[:, 0:n_column - 1]
    y_test = data_test.iloc[:, -1]

    return x_train, y_train, x_test, y_test


def encoder(data) -> object:
    coder = LabelEncoder()
    coder.fit(data)
    return coder.transform(data)


def training(
        x_train, x_test,
        y_train, y_test,
        n_neighbors: int
) -> float:
    model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance'
    )
    model.fit(x_train, y_train)

    return model.score(x_test, y_test)


def plot_data(data) -> None:
    colors = ['red' if c == 'red' else 'green' for c in data['Colors']]

    plt.scatter(data['X1'], data['X2'], c=colors)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.title('KNN part 3 initial data')
    plt.show()


def plot_score(x_data, y_data) -> None:
    plt.plot(x_data, y_data)
    plt.ylabel('accuracy')
    plt.xlabel('number of neighbours')
    plt.title('KNN part 3 accuracy')
    plt.grid(True)
    plt.show()


data_train = read_csv(
        filepath_or_buffer='svmdata4.txt',
        sep='	', header=0
)
data_test = read_csv(
        filepath_or_buffer='svmdata4test.txt',
        sep='	', header=0
)

plot_data(data_train)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

n_neighbors_list = range(1, 52, 2)
score_list = []
for n_neighbors in n_neighbors_list:
    score = training(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_neighbors=n_neighbors
    )
    score_list.append(score)

plot_score(x_data=n_neighbors_list, y_data=score_list)
print(f'max accuracy {max(score_list)}')
