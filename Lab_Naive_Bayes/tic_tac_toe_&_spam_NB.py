import matplotlib.pyplot as plt

from pandas import read_csv
from numpy import arange
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB


class CategoricalNBLearn:
    def __init__(self, ):
        self.random_state = 100
        self.train_size = arange(0.01, 1., 0.001)

    def encoder(self, data, flag):  # create labels for categorical data
        if flag:
            coder = OrdinalEncoder()
        else:
            coder = LabelEncoder()
        coder.fit(data)
        return coder.transform(data)

    def data_segregate(self, data, flag):
        n_column = data.shape[1]
        if flag:
            return data[:, 0:n_column - 1], data[:, -1]
        else:
            return data.iloc[:, 0:n_column - 1], data.iloc[:, -1]

    # fit of model and take the score(accuracy in this one)
    def model_score(self, x_train, y_train, x_test, y_test, flag):
        if flag:
            model = CategoricalNB()
        else:
            model = GaussianNB()
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    # plot learned score
    def plot(self, x_data, y_data, x_label, y_label, title):
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def run(self, data, title, flag):  # run algorithm
        if flag:
            data = self.encoder(data, flag)
            x, y = self.data_segregate(data, flag)
        else:
            x, y = self.data_segregate(data, flag)
            y = self.encoder(y, flag)

        score_list = []

        # divide the data into test and train
        for size in self.train_size:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                train_size=size,
                                                                random_state=self.random_state,
                                                                shuffle=True)
            score = self.model_score(x_train, y_train, x_test, y_test, flag)
            score_list.append(score)

        self.plot(self.train_size, score_list,
                  'Train size', 'Accuracy', title)


# import data
data_tic_tac_toe = read_csv('Tic_tac_toe.txt', sep=",", header=None)

# use NB for classification
CategoricalNBLearn().run(data=data_tic_tac_toe, title='Tic-tac-toe Naive Bayes classifier', flag=True)

data_spam = read_csv('spam.txt', sep=",", header=0)
CategoricalNBLearn().run(data=data_spam, title='Spam Naive Bayes classifier', flag=False)
