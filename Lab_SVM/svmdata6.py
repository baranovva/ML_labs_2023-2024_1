import numpy as np

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from functions_svm import training, encoder, line_plot, data_split_task_3

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
