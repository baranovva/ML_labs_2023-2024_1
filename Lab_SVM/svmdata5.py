from functions_svm import data_split, training, encoder, visualisation, grid_search_task_5, reader

data_train = reader(file_name='svmdata5.txt')
data_test = reader(file_name='svmdata5test.txt')

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

best_parameters = grid_search_task_5(x_train, y_train)

score_train, score_test, model = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        kernel_name=best_parameters['kernel'],
        C_for_C_support=best_parameters['C'],
        gamma=best_parameters['gamma'],
        degree=best_parameters['degree']
)

print(score_train, score_test)

visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        model=model,
        task_name='5',
        colors='red,green'
)

# hyperparametr with overfitting
gamma = 50

score_train, score_test, model = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        kernel_name='poly',
        gamma=gamma,
        degree=2
)

print(score_train, score_test)

visualisation(
        x_data=x_train.to_numpy(),
        y_data=y_train,
        model=model,
        task_name='5 overfitting train',
        colors='red,green'
)

visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        model=model,
        task_name='5 overfitting test',
        colors='red,green'
)
