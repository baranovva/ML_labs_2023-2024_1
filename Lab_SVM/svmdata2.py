from functions_svm import data_split, training, encoder, visualisation, grid_search_task_2, reader

data_train = reader(file_name='svmdata2.txt')
data_test = reader(file_name='svmdata2test.txt')

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

kwards = {'kernel': 'linear', 'degree': 3, 'gamma': 'scale'}
best_parameters = grid_search_task_2(x_train, y_train, **kwards)

score_train, score_test, model = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        kernel_name='linear',
        C_for_C_support=best_parameters['C'],
)

print(score_train, score_test)

visualisation(
        x_data=x_train.to_numpy(),
        y_data=y_train,
        model=model,
        task_name='2 train data',
        colors='red,green'
)
visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        model=model,
        task_name='2 test data',
        colors='red,green'
)
