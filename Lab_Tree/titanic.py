import numpy as np

from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class TitanicKNN:
    def __init__(self):
        self.random_state = 2023

    def data_preparing(self, file_name: str) -> object:  # filling missing data and del some data classes
        data = read_csv(file_name)

        mid_age = data['Age'].median()
        data['Age'] = data['Age'].fillna(mid_age)

        data = data.drop(
                columns=[
                    'PassengerId', 'Name', 'Ticket',
                    'Cabin', 'Embarked', 'Fare'
                ]
        )
        return data

    # separate data as input and output
    def data_separate(self, data) -> object:
        x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
        y = data['Survived']
        return x, y

    # create labels for categorical data
    def encoder(self, data) -> object:
        coder = LabelEncoder()
        coder.fit_transform(data)
        return coder.transform(data)

    # fit model and take predict
    def model_predict(self,
                      x_train, y_train,
                      x_test, y_test,
                      criterion,
                      max_depth,
                      min_samples_split,
                      min_samples_leaf
                      ) -> object:
        model = DecisionTreeClassifier(
                random_state=self.random_state,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
        )
        model.fit(x_train, y_train)

        return model.score(x_test, y_test)

    def grid_search(self, x_train, y_train) -> object:
        tree = DecisionTreeClassifier(random_state=self.random_state)
        param_grid = {
            'criterion': ('gini', 'entropy', 'log_loss'),
            'max_depth': np.arange(2, 6),
            'min_samples_split': np.arange(2, 7),
            'min_samples_leaf': np.arange(1, 6)
        }

        grid_search = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        print("The best:", grid_search.best_params_)
        return grid_search.best_params_


KNN = TitanicKNN()
# preparing data (train & test)
train_data = KNN.data_preparing(file_name='train.csv')
train_data['Sex'] = KNN.encoder(data=train_data['Sex'])
x_train, y_train = KNN.data_separate(data=train_data)

test_data = KNN.data_preparing(file_name='test.csv')
test_data['Sex'] = KNN.encoder(data=test_data['Sex'])
x_test = test_data

best_parameters = KNN.grid_search(x_train, y_train)

submission_data = read_csv(
        'gender_submission.csv',
        sep=",", header=0
)
y_test = submission_data['Survived']

# use model for prediction
score = KNN.model_predict(
        x_train, y_train,
        x_test, y_test,
        criterion=best_parameters['criterion'],
        max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf']
)
print(score)
