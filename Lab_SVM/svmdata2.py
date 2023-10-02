from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR


def data_split(data_train, data_test) -> object:
    n_column = data_train.shape[1]

    x_train = data_train.iloc[:, 0:n_column - 1]
    y_train = data_train.iloc[:, -1]

    x_test = data_test.iloc[:, 0:n_column - 1]
    y_test = data_test.iloc[:, -1]

    return x_train, y_train, x_test, y_test


def training(
        x_train,
        x_test,
        y_train,
        y_test,
        model_name: str,
        kernel_name: str,
        C_for_C_support: float,
        epsilon_for_E_support: float,
        gamma
) -> float:
    if model_name == 'C-Support':
        model = SVC(
                kernel=kernel_name,
                C=C_for_C_support,
                gamma=gamma
        )
    elif model_name == 'Epsilon-Support':
        model = SVR(
                kernel=kernel_name,
                gamma=gamma,
                epsilon=epsilon_for_E_support
        )
    else:
        print('invalid model name')

    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def encoder(data) -> object:
    coder = LabelEncoder()
    coder.fit(data)
    return coder.transform(data)


data_train = read_csv(
        'svmdata1.txt',
        sep="	", header=0
)
data_test = read_csv(
        'svmdata1.txt',
        sep="	", header=0
)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

score = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_name='C-Support',  # C/Epsilon
        kernel_name='linear',
        C_for_C_support=1,
        epsilon_for_E_support=None,
        gamma=1
)

print(score)
