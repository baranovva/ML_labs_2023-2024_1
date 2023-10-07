from pandas import read_csv
from functions_svm import data_split, training, encoder, visualisation, grid_search_task_2

data_train = read_csv(
        filepath_or_buffer='svmdata2.txt',
        header=0,
        sep='	'
)

data_test = read_csv(
        filepath_or_buffer='svmdata2test.txt',
        header=0,
        sep='	'
)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

best_parameters = grid_search_task_2(
        x_train=x_train,
        y_train=y_train,
        kernel_name='linear',
        gamma='scale',  # default
        degree=3  # default
)

score_train, score_test, model, __ = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_name='C-Support',  # C/Epsilon
        kernel_name='linear',
        C_for_C_support=best_parameters['C'],
        epsilon_for_E_support=None,
        gamma='scale',  # default
        degree=3  # default
)

print(score_train, score_test)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
visualisation(
        x_data=x_train,
        y_data=y_train,
        x_test=None,
        model=model,
        task_name='task 2 train data',
        colors='red,green'
)
visualisation(
        x_data=x_test,
        y_data=y_test,
        x_test=None,
        model=model,
        task_name='task 2 test data',
        colors='red,green'
)
