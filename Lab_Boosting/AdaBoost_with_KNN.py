import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def split_data(data, train_size=0.9, random_state=2023):
    def encoder(data) -> object:
        coder = LabelEncoder()
        coder.fit_transform(data)
        return coder.transform(data)

    n_column = data.shape[1]
    x = data.iloc[:, 0:n_column - 1]
    y = data.iloc[:, -1]

    y = encoder(data=y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state,
                                                        shuffle=True)
    return x_train, x_test, y_train, y_test


class AdaBoost:
    def __init__(self, n_estimators=1):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        sample_weights = np.ones(X.shape[0])
        for _ in range(self.n_estimators):
            X = X * sample_weights[:, np.newaxis]

            model = KNeighborsClassifier()
            model.fit(X, y)
            self.models.append(model)

            y_pred = model.predict(X)

            err = np.sum(sample_weights * (y != y_pred)) / np.sum(sample_weights)

            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10)) + np.log(2)
            self.alphas.append(alpha)
            if err >= 0.5:
                break

            sample_weights *= np.exp(-alpha * y * y_pred)
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.models[0].classes_)))
        for alpha, model in zip(self.alphas, self.models):
            alpha = alpha / np.sum(self.alphas)
            scores += alpha * model.predict_proba(X)

        return np.argmax(scores, axis=1)


data_glass = read_csv(filepath_or_buffer='../Lab_Boosting/vehicle.csv', sep=',', header=0)
x_train, x_test, y_train, y_test = split_data(data_glass)

adaboost = AdaBoost(n_estimators=1)
adaboost.fit(x_train.to_numpy(), y_train)
y_pred = adaboost.predict(x_test.to_numpy())
print(np.mean(y_pred == y_test))
