from pandas import read_csv
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data = read_csv('cygage.txt', sep='\t', header=0)

data_x = data.iloc[:, 1:]
data_y = data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.9, random_state=2023)

lin_reg = LinearRegression(n_jobs=-1).fit(x_train, y_train)
ridge_reg = Ridge(alpha=0.5).fit(x_train, y_train)
lasso_reg = Lasso(alpha=0.5).fit(x_train, y_train)
elastic_reg = ElasticNet(alpha=0.5).fit(x_train, y_train)

print('MAPE lin:', mean_absolute_percentage_error(y_test, lin_reg.predict(x_test)))
print('MAPE ridge:', mean_absolute_percentage_error(y_test, ridge_reg.predict(x_test)))
print('MAPE lasso:', mean_absolute_percentage_error(y_test, lasso_reg.predict(x_test)))
print('MAPE elastic:', mean_absolute_percentage_error(y_test, elastic_reg.predict(x_test)))
