import numpy as np
import matplotlib.pyplot as plt

from numpy import arange
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

np.random.seed(2023)


# generate data as points
def data_generator(n_point, mean, covariance, class_name):
    covar = np.array([[covariance, 0], [0, covariance]])
    x_data = np.random.multivariate_normal(mean, covar, n_point)
    y_data = np.full(n_point, class_name)
    return x_data, y_data


class GaussianNBLearn:
    def __init__(self):
        self.random_state = 100
        self.train_size = arange(0.02, 1., 0.02)  # create array with train size

    # fit of model and take the score(accuracy in this one)
    def model_score(self, x_train, y_train, x_test, y_test):
        model = GaussianNB()
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)

    # plot learned score
    def plot(self, x_data, y_data, x_label, y_label, title):
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    # run algorithm
    def run(self, x_data, y_data, title):
        score_list = []

        # divide the data into test and train
        for size in self.train_size:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                                train_size=size,
                                                                random_state=self.random_state,
                                                                shuffle=True)
            score = self.model_score(x_train, y_train, x_test, y_test)
            score_list.append(score)

        self.plot(self.train_size, score_list,
                  'Train size', 'Accuracy', title)


n_point = 50  # number of points for all classes

mean_class_1 = [10, 14]  # mean for class -1
covariance_class_1 = 4  # covariance for class - 1
x1, y1 = data_generator(n_point, mean_class_1, covariance_class_1, -1)

mean_class_2 = [20, 18]
covariance_class_2 = 3
x2, y2 = data_generator(n_point, mean_class_2, covariance_class_2, 1)

# plot generated data
plt.scatter(x1[:, 0], x1[:, 1], color='blue', label='Class -1')
plt.scatter(x2[:, 0], x2[:, 1], color='red', label='Class 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data distribution')
plt.legend()
plt.grid(True)
plt.show()

# concatenate data
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

# use NB for classification
GaussianNBLearn().run(x_data=x, y_data=y, title='Distribution Naive Bayes classifier')
