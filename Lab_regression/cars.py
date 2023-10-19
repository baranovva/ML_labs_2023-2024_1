from numpy import array
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

data = read_csv('cars.csv', sep=',', header=0)

data_x = data.iloc[:, :1]
data_y = data.iloc[:, 1]

lin_reg = LinearRegression(n_jobs=-1).fit(data_x, data_y)
ridge_reg = Ridge(alpha=0.5).fit(data_x, data_y)
lasso_reg = Lasso(alpha=0.5).fit(data_x, data_y)
elastic_reg = ElasticNet(alpha=0.5).fit(data_x, data_y)

print('RMSE lin:', mean_squared_error(data_y, lin_reg.predict(data_x), squared=False))
print('RMSE ridge:', mean_squared_error(data_y, ridge_reg.predict(data_x), squared=False))
print('RMSE lasso:', mean_squared_error(data_y, lasso_reg.predict(data_x), squared=False))
print('RMSE elastic:', mean_squared_error(data_y, elastic_reg.predict(data_x), squared=False))

vel = DataFrame(array([40]), columns=['speed'])

print(*lin_reg.predict(vel))
print(*ridge_reg.predict(vel))
print(*lasso_reg.predict(vel))
print(*elastic_reg.predict(vel))
