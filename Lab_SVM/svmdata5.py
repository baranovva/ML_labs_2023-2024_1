from pandas import read_csv
from functions_svm import data_split, training, encoder, visualisation, grid_search_task_5

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
