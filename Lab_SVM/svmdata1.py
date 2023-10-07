from pandas import read_csv
from functions_svm import data_split, training, encoder, visualisation

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

visualisation(
        x_data=x_test.to_numpy(),
        y_data=y_test,
        x_test=x_train.to_numpy(),
        model=model,
        task_name='task 1',
        colors='green,red'
)
