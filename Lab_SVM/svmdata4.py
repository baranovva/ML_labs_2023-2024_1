from pandas import read_csv
from functions_svm import data_split, training, encoder, grid_search_task_4

data_train = read_csv(
        filepath_or_buffer='svmdata4.txt',
        header=0,
        sep='	'
)

data_test = read_csv(
        filepath_or_buffer='svmdata4test.txt',
        header=0,
        sep='	'
)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

best_parameters = grid_search_task_4(x_train, y_train)

score_train, score_test, __, __ = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_name='C-Support',  # C/Epsilon
        kernel_name=best_parameters['kernel'],
        C_for_C_support=best_parameters['C'],
        epsilon_for_E_support=None,
        gamma=best_parameters['gamma'],
        degree=best_parameters['degree']
)

print(score_train, score_test)
