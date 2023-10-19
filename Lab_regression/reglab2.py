from itertools import combinations
from pandas import read_csv
from sklearn.linear_model import LinearRegression

data = read_csv('reglab2.txt', sep='\t', header=0)

data_x = data.iloc[:, 1:]
data_y = data.iloc[:, 0]

columns = list(combinations(data_x.columns, 2)) + list(combinations(data_x.columns, 3))
for col_names in columns:
    lin_reg = LinearRegression(n_jobs=-1).fit(data_x[list(col_names)], data_y)
    print(*col_names)
    print(lin_reg.score(data_x[list(col_names)], data_y))

lin_reg = LinearRegression(n_jobs=-1).fit(data_x, data_y)
print(*data_x.columns)
print(lin_reg.score(data_x, data_y))
