from functions_svm import data_split, training, encoder, grid_search_task_4, reader

data_train = reader(file_name='svmdata4.txt')
data_test = reader(file_name='svmdata4test.txt')

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

best_parameters = grid_search_task_4(x_train, y_train)

score_train, score_test, __ = training(
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
