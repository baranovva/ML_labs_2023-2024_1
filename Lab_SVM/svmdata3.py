import numpy as np

from functions_svm import training, encoder, reader
from functions_svm import grid_search_task_2, data_split_task_3, line_plot, bar_plot

data = reader(file_name='svmdata3.txt')

(x_train, x_test,
 y_train, y_test) = data_split_task_3(
        data=data,
        train_size=0.5
)

y_test = encoder(y_test)
y_train = encoder(y_train)

kernel_list = ["poly", "rbf", "sigmoid"]
score_test_list = []

for kernel in kernel_list:
    if kernel != "poly":
        kwards = {'kernel': kernel, 'degree': 3, 'gamma': 'scale'}
        best_parameters = grid_search_task_2(x_train, y_train, **kwards)

        __, score_test, __ = training(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                kernel_name=kernel,
                C_for_C_support=best_parameters['C'],
        )
        score_test_list.append(score_test)

    else:
        poly_degree_list = np.arange(1, 11, 1)
        score_test_list_poly = []

        for degree in poly_degree_list:
            kwards = {'kernel': kernel, 'degree': degree, 'gamma': 'scale'}
            best_parameters = grid_search_task_2(x_train, y_train, **kwards)

            __, score_test, __ = training(
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    kernel_name=kernel,
                    C_for_C_support=best_parameters['C'],
                    degree=degree
            )
            score_test_list_poly.append(score_test)

        score_test_list.append(max(score_test_list_poly))

print(score_test_list)

line_plot(
        x_data=poly_degree_list,
        y_data=score_test_list_poly,
        x_label='poly degree',
        title='SVM task 3 poly kernel'
)

bar_plot(x_data=kernel_list, y_data=score_test_list)
