import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from mlxtend.plotting import plot_decision_regions


def data_split(data_train: object, data_test: object) -> object:
    n_column = data_train.shape[1]

    x_train = data_train.iloc[:, 0:n_column - 1]
    y_train = data_train.iloc[:, -1]

    x_test = data_test.iloc[:, 0:n_column - 1]
    y_test = data_test.iloc[:, -1]

    return x_train, y_train, x_test, y_test


def training(
        x_train: object,
        x_test: object,
        y_train: object,
        y_test: object,
        model_name: str,
        kernel_name: str,
        C_for_C_support: float,
        epsilon_for_E_support: float,
        gamma: object,
        degree: int
) -> object:
    if model_name == 'C-Support':
        model = SVC(
                kernel=kernel_name,
                C=C_for_C_support,
                gamma=gamma,
                degree=degree
        )
    elif model_name == 'Epsilon-Support':
        model = SVR(
                kernel=kernel_name,
                gamma=gamma,
                epsilon=epsilon_for_E_support,
                degree=degree
        )
    else:
        print('invalid model name')

    model.fit(x_train, y_train)
    return model.score(x_train, y_train), model.score(x_test, y_test), model, len(model.support_vectors_)


def encoder(data: object) -> object:
    coder = LabelEncoder()
    coder.fit(data)
    return coder.transform(data)


def visualisation(
        x_data: object,
        y_data: object,
        x_test: object,
        model: object,
        task_name: str,
        colors: str
) -> None:
    p = plot_decision_regions(
            x_data,
            y_data,
            clf=model,
            legend=2,
            zoom_factor=5,
            X_highlight=x_test,
            colors=colors
    )
    handles, labels = p.get_legend_handles_labels()
    colors_list = list(map(str, colors.split(',')))
    p.legend(
            handles,
            [colors_list[0], colors_list[1]],
            framealpha=0.3,
            scatterpoints=1
    )

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('SVM for ' + task_name)

    plt.show()


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

del (data_test, data_train, x_test, x_train, y_test, y_train,
     score_train, score_test, model, n_support_vectors)
'''


# task 2

def grid_search_task_2(
        x_train: object,
        y_train: object,
        kernel_name: str,
        gamma: object
) -> object:
    model = SVC(
            kernel=kernel_name,
            gamma=gamma
    )

    parameters = {'C': np.arange(0.01, 10, 0.01)}

    grid_search = GridSearchCV(model, parameters, n_jobs=8)
    grid_search.fit(x_train, y_train)

    print("The best:", grid_search.best_params_)
    return grid_search.best_params_


'''
data_train = read_csv(
        filepath_or_buffer='svmdata2.txt',
        header=0,
        sep='	'
)

data_test = read_csv(
        filepath_or_buffer='svmdata2test.txt',
        header=0,
        sep='	'
)

(x_train, y_train,
 x_test, y_test) = data_split(data_train, data_test)

y_test = encoder(y_test)
y_train = encoder(y_train)

best_parameters = grid_search_task_2(
        x_train=x_train,
        y_train=y_train,
        kernel_name='linear',
        gamma=3  # default
)

score_train, score_test, model, n_support_vectors = training(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_name='C-Support',  # C/Epsilon
        kernel_name='linear',
        C_for_C_support=best_parameters['C'],
        epsilon_for_E_support=None,
        gamma='scale',  # default
        degree=3  # default
)

print(score_train, score_test)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
visualisation(
        x_data=x_train,
        y_data=y_train,
        x_test=None,
        model=model,
        task_name='task 2 train data',
        colors='red,green'
)
visualisation(
        x_data=x_test,
        y_data=y_test,
        x_test=None,
        model=model,
        task_name='task 2 test data',
        colors='red,green'
)

del (data_train, data_test, x_train, y_train, x_test, y_test, best_parameters,
     score_train, score_test, model, n_support_vectors)
'''

# task 3

def data_split_task_3(
        data: object,
        train_size: float,
        random_state: int
) -> object:
    n_column = data.shape[1]

    x_data = data.iloc[:, 0:n_column - 1]
    y_data = data.iloc[:, -1]

    (x_train, x_test,
     y_train, y_test) = train_test_split(
            x_data, y_data,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
    )

    return x_train, x_test, y_train, y_test

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

kernel_list = [
    "polynomial",
    "radial",
    "sigmoid"
]
