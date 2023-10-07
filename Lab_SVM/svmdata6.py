import numpy as np

from sklearn.metrics import mean_squared_error
from functions_svm import training, encoder, line_plot, data_split_task_3, reader

data = reader(file_name='svmdata6.txt')

(x_train, x_test,
 y_train, y_test) = data_split_task_3(
        data=data,
        train_size=0.5
)

y_test = encoder(y_test)
y_train = encoder(y_train)

score_test_list = []
epsilon_list = np.arange(0, 2, 0.01)

for epsilon in epsilon_list:
    __, __, model = training(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            model_name='Epsilon-Support',  # C/Epsilon
            kernel_name="rbf",
            epsilon_for_E_support=epsilon,
    )
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    score_test_list.append(mse)

line_plot(
        x_data=epsilon_list,
        y_data=score_test_list,
        x_label='epsilon',
        y_label='MSE',
        title='SVM task 6 '
)
