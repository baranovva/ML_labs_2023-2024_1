import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from mlxtend.plotting import plot_decision_regions

from functions_svm import data_split, training, encoder, visualisation

# task 1
'''
data_train = read_csv(
        'svmdata1.txt',
        sep="	", header=0
)
data_test = read_csv(
        'svmdata1test.txt',
        sep="	", header=0
)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

score_train, score_test, model, n_support_vectors = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_name='C-Support',  # C/Epsilon
        kernel_name='linear',
        C_for_C_support=1,
        epsilon_for_E_support=None,
        gamma='scale',  # default
        degree=3  # default
)

print(score_test, n_support_vectors)

x_test = x_test.to_numpy()
visualisation(
        x_data=x_test,
        y_data=y_test,
        x_test=x_train.to_numpy(),
        model=model,
        task_name='task 1',
        colors='green,red'
)
'''




# task 3
'''
from functions_svm import data_split_task_3, line_plot, bar_plot

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

            score_train, score_test, model, __ = training(
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

bar_plot(kernel_list, score_test_list)'''

# task 4
'''
from functions_svm import grid_search_task_4

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
'''
# task 5
from functions_svm import grid_search_task_5


data_train = read_csv(
        filepath_or_buffer='svmdata5.txt',
        header=0,
        sep='	'
)

data_test = read_csv(
        filepath_or_buffer='svmdata5test.txt',
        header=0,
        sep='	'
)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

best_parameters = grid_search_task_5(x_train, y_train)

score_train, score_test, model, n_support_vectors = training(
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


visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        x_test=None,
        model=model,
        task_name='task 5',
        colors='red,green'
)

# hyperparametr with overfitting
gamma = 50

score_train, score_test, model, __ = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_name='C-Support',  # C/Epsilon
        kernel_name='poly',
        C_for_C_support=1,
        epsilon_for_E_support=None,
        gamma=gamma,
        degree=2
)

print(score_train, score_test)

visualisation(
        x_data=x_train.to_numpy(),
        y_data=y_train,
        x_test=None,
        model=model,
        task_name='task 4 train',
        colors='red,green'
)

visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        x_test=None,
        model=model,
        task_name='task 4 test',
        colors='red,green'
)

# task 6
'''
data = read_csv(
        filepath_or_buffer='svmdata6.txt',
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

score_test_list = []

epsilon_list = np.arange(0, 2, 0.01)

for epsilon in epsilon_list:
    __, __, model, __ = training(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            model_name='Epsilon-Support',  # C/Epsilon
            kernel_name="rbf",
            C_for_C_support=1,  # default
            epsilon_for_E_support=epsilon,
            gamma='scale',  # default
            degree=3  # default
    )
    y_pred = model.predict(x_test)
    score_test_list.append(mean_squared_error(y_true=y_test, y_pred=y_pred))

line_plot(
        x_data=epsilon_list,
        y_data=score_test_list,
        x_label='epsilon',
        y_label='accuracy for test',
        title='SVM task 6 '
)
'''
