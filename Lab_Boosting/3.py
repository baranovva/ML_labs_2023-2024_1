from pandas import read_csv
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
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


for name in ('vehicle', 'Glass'):
    print(name)
    data = read_csv(filepath_or_buffer='../Lab_Boosting/' + name + '.csv', sep=',', header=0)

    x_train, x_test, y_train, y_test = split_data(data)

    knn = KNeighborsClassifier()
    model = BaggingClassifier(random_state=2023, n_estimators=50, estimator=knn).fit(x_train, y_train)
    print('Ada:', model.score(x_test, y_test))

    knn.fit(x_train, y_train)
    print('Knn:', knn.score(x_test, y_test))
