import matplotlib.pyplot as plt

from numpy import argmin
from math import pow
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = read_csv('longley.csv', sep=',', header=0)

data_x = data.drop(['Employed', 'Population'], axis=1)
data_y = data['Employed']

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.5, random_state=2023)

alphas = []
train_errors = []
test_errors = []

for i in range(0, 26):
    alpha = pow(10, 3 + 0.2 * i)
    ridge_reg = Ridge(alpha=alpha).fit(x_train, y_train)

    alphas.append(alpha)
    train_errors.append(mean_squared_error(y_train, ridge_reg.predict(x_train), squared=False))
    test_errors.append(mean_squared_error(y_test, ridge_reg.predict(x_test), squared=False))

plt.plot(alphas, train_errors, label='train')
plt.plot(alphas, test_errors, label='test')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

best_alpha = alphas[argmin(test_errors)]
print('Оптимальное alpha:', best_alpha)

ridge_reg = Ridge(alpha=best_alpha).fit(x_train, y_train)
print('RMSE', mean_squared_error(y_test, ridge_reg.predict(x_test), squared=False))
