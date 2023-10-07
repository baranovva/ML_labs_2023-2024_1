from functions_svm import data_split, training, encoder, visualisation, reader

data_train = reader(file_name='svmdata1.txt')
data_test = reader(file_name='svmdata1test.txt')

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

__, score_test, model = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        kernel_name='linear'
)

n_support_vectors = len(model.support_vectors_)
print(score_test, n_support_vectors)

visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        x_test=x_train.to_numpy(),
        model=model,
        task_name='1',
)
