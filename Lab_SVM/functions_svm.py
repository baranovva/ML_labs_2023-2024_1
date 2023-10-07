import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from sklearn.metrics import mean_squared_error
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


def grid_search_task_2(
        x_train: object,
        y_train: object,
        kernel_name: str,
        gamma: object,
        degree: int
) -> object:
    model = SVC(
            kernel=kernel_name,
            gamma=gamma,
            degree=degree
    )

    parameters = {'C': np.arange(0.01, 10, 0.01)}

    grid_search = GridSearchCV(model, param_grid=parameters, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    print("The best:", grid_search.best_params_, 'with score:', grid_search.best_score_)
    return grid_search.best_params_


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


def line_plot(
        x_data: object,
        y_data: object,
        x_label: str,
        y_label: str,
        title: str
) -> None:
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


def bar_plot(x_data, y_data):
    plt.bar(x_data, y_data)
    plt.ylabel('accuracy for test')
    plt.title('SVM task 3 all kernels')
    plt.show()


def grid_search_task_4(x_data: object, y_data: object) -> object:
    model = SVC()

    parameters = {
        'C': np.arange(1, 10, 1),
        'kernel': ("poly", "rbf", "sigmoid"),
        'degree': np.arange(1, 5, 1),
        'gamma': ('scale', 'auto')
    }

    grid_search = GridSearchCV(model, param_grid=parameters, n_jobs=-1)
    grid_search.fit(x_data, y_data)

    print("The best:", grid_search.best_params_, 'with score:', grid_search.best_score_)
    return grid_search.best_params_


def grid_search_task_5(x_data: object, y_data: object) -> object:
    model = SVC()

    parameters = {
        'C': np.arange(-5, 6, 1),
        'kernel': ("poly", "rbf", "sigmoid"),
        'degree': np.arange(1, 5, 1),
        'gamma': np.arange(0, 15, 1)
    }

    grid_search = GridSearchCV(model, param_grid=parameters, n_jobs=-1)
    grid_search.fit(x_data, y_data)

    print("The best:", grid_search.best_params_, 'with score:', grid_search.best_score_)
    return grid_search.best_params_