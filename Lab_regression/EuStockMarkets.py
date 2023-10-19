from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = read_csv('EuStockMarkets.csv', sep=',', header=0)

data_x = data.iloc[:, 1:]
data_y = data.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.9, random_state=2023)

plt.figure(figsize=(10, 6))
for column in data_x.columns:
    plt.plot(data_x[column], label=column)
plt.legend()
plt.xlabel('Day')
plt.ylabel('Stock Prices')
plt.grid(True)
plt.show()

reg_all = LinearRegression(n_jobs=-1).fit(x_train, y_train)
print('RMSE all:', mean_squared_error(y_test, reg_all.predict(x_test), squared=False))

for column in x_train.columns:
    x_train_drop = x_train[column].values.reshape(-1, 1)
    x_test_drop = x_test[column].values.reshape(-1, 1)
    reg_drop = LinearRegression(n_jobs=-1).fit(x_train_drop, y_train)

    rmse = mean_squared_error(y_test, reg_drop.predict(x_test_drop), squared=False)
    print(f'RMSE {column}: {rmse}')
    print(f'R^2 {column}: {reg_drop.coef_[0]}')
