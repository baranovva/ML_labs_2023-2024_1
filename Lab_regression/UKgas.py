import matplotlib.pyplot as plt

from numpy import mean
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = read_csv('UKgas.csv', sep=',', header=0)

data_x = data.iloc[:, :1]
data_y = data.iloc[:, 1]

data_x_1 = data_x.iloc[lambda x: x.index % 4 == 0]
data_x_2 = data_x.iloc[lambda x: x.index % 4 == 1]
data_x_3 = data_x.iloc[lambda x: x.index % 4 == 2]
data_x_4 = data_x.iloc[lambda x: x.index % 4 == 3]

data_y_1 = data_y.iloc[lambda x: x.index % 4 == 0]
data_y_2 = data_y.iloc[lambda x: x.index % 4 == 1]
data_y_3 = data_y.iloc[lambda x: x.index % 4 == 2]
data_y_4 = data_y.iloc[lambda x: x.index % 4 == 3]

plt.plot(data_x, data_y, label='all quarters')
plt.plot(data_x_1, data_y_1, label='1 quarter')
plt.plot(data_x_2, data_y_2, label='2 quarter')
plt.plot(data_x_3, data_y_3, label='3 quarter')
plt.plot(data_x_4, data_y_4, label='4 quarter')
plt.xlabel('Quarter')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)
plt.show()

lin_reg = LinearRegression(n_jobs=-1)

lin_reg_all = lin_reg.fit(data_x, data_y)
print('RMSE all quarters:', mean_squared_error(data_y, lin_reg_all.predict(data_x), squared=False))

lin_reg_1 = lin_reg.fit(data_x_1, data_y_1)
print('RMSE 1 quarters:', mean_squared_error(data_y_1, lin_reg_1.predict(data_x_1), squared=False))

lin_reg_2 = lin_reg.fit(data_x_2, data_y_2)
print('RMSE 2 quarters:', mean_squared_error(data_y_2, lin_reg_2.predict(data_x_2), squared=False))

lin_reg_3 = lin_reg.fit(data_x_3, data_y_3)
print('RMSE 3 quarters:', mean_squared_error(data_y_3, lin_reg_3.predict(data_x_3), squared=False))

lin_reg_4 = lin_reg.fit(data_x_4, data_y_4)
print('RMSE 4 quarters:', mean_squared_error(data_y_4, lin_reg_4.predict(data_x_4), squared=False))

quarters = data_x.shape[0] + (2016 - 1986) * 4 - 3
year_2016 = DataFrame(([quarters, ], [quarters + 1, ], [quarters + 2, ], [quarters + 3, ]), columns=['quarter'])

print(mean(lin_reg_all.predict(year_2016)))
print(mean(lin_reg_1.predict(year_2016.iloc[:1, :])))
print(mean(lin_reg_2.predict(year_2016.iloc[1:2, :])))
print(mean(lin_reg_3.predict(year_2016.iloc[2:3, :])))
print(mean(lin_reg_4.predict(year_2016.iloc[3:4, :])))
