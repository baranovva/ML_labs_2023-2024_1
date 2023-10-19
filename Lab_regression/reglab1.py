from pandas import read_csv
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = read_csv('reglab1.txt', sep='\t')

data_x = data.iloc[:, 1:]
data_y = data.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.9, random_state=2023)

lin_reg = LinearRegression(n_jobs=-1).fit(x_train, y_train)
ridge_reg = Ridge(alpha=0.5).fit(x_train, y_train)
lasso_reg = Lasso(alpha=0.5).fit(x_train, y_train)
elastic_reg = ElasticNet(alpha=0.5).fit(x_train, y_train)

print('RMSE lin:', mean_squared_error(y_test, lin_reg.predict(x_test), squared=False))
print('RMSE ridge:', mean_squared_error(y_test, ridge_reg.predict(x_test), squared=False))
print('RMSE lasso:', mean_squared_error(y_test, lasso_reg.predict(x_test), squared=False))
print('RMSE elastic:', mean_squared_error(y_test, elastic_reg.predict(x_test), squared=False))
