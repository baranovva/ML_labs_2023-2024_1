import numpy as np

from pandas import read_csv
from functions_svm import training, encoder
from functions_svm import grid_search_task_2, data_split_task_3, line_plot, bar_plot

data = read_csv(
        filepath_or_buffer='svmdata3.txt',
        header=0,
        sep='	'
)

(x_train, x_test,
 y_train, y_test) = data_split_task_3(
        data=data,
        train_size=0.5,
        random_state=2023
)

y_test = encoder(y_test)
y_train = encoder(y_train)

kernel_list = [
    "linear",
    "poly",
    "rbf",
    "sigmoid"
]
score_test_list = []

for kernel in kernel_list:
    if kernel != "poly":
        best_parameters = grid_search_task_2(
                x_train=x_train,
                y_train=y_train,
                kernel_name=kernel,
                gamma='scale',  # default
                degree=3  # default
        )

        __, score_test, __, __ = training(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                model_name='C-Support',  # C/Epsilon
                kernel_name=kernel,
                C_for_C_support=best_parameters['C'],
                epsilon_for_E_support=None,
                gamma='scale',  # default
                degree=3  # default
        )
        score_test_list.append(score_test)

    else:
        poly_degree_list = np.arange(1, 11, 1)
        score_test_list_poly = []

        for degree in poly_degree_list:
            best_parameters = grid_search_task_2(
                    x_train=x_train,
                    y_train=y_train,
                    kernel_name=kernel,
                    gamma='scale',  # default
                    degree=degree
            )

            __, score_test, __, __ = training(
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_name='C-Support',  # C/Epsilon
                    kernel_name=kernel,
                    C_for_C_support=best_parameters['C'],
                    epsilon_for_E_support=None,
                    gamma='scale',  # default
                    degree=degree
            )
            score_test_list_poly.append(score_test)

        score_test_list.append(max(score_test_list_poly))

print(score_test_list)

line_plot(
        x_data=poly_degree_list,
        y_data=score_test_list_poly,
        x_label='poly degree',
        y_label='accuracy for test',
        title='SVM task 3 poly kernel'
)

bar_plot(x_data=kernel_list, y_data=score_test_list)
