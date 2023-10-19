import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = read_csv('sunspot_year.csv', sep=',', header=0)

X = data['year'].values.reshape(-1, 1)
y = data['x'].values.reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('RMSE lin:', mean_squared_error(y, reg.predict(X), squared=False))

plt.plot(data['year'], data['x'], label='Initial data')
plt.plot(data['year'], reg.predict(X), label='linear regression')
plt.xlabel('Year')
plt.ylabel('Number of sunspots')
plt.legend()
plt.grid(True)
plt.show()
