from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from numpy.random import seed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

seed(2023)


class TitanicKNN:
    def data_preparing(file_name):  # filling missing data and del some data classes
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

    def encoder(data):  # create labels for categorical data
        coder = LabelEncoder()
        coder.fit_transform(data)
        return coder.transform(data)

    # separate data as input and output
    def data_separate(data):
        x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
        y = data['Survived']
        return x, y

    # fit model and take predict
    def model_predict(
            x_train, y_train,
            x_test,
            n_neighbors: int
    ) -> object:
        model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights='distance'
        )
        model.fit(x_train, y_train)

        return model.predict(x_test)

    def grid_search(x_train, y_train) -> object:
        model = KNeighborsClassifier(weights='distance')
        parameters = {'n_neighbors': range(1, 25, 2)}

        grid_search = GridSearchCV(model, parameters)
        grid_search.fit(x_train, y_train)

        print("The best:", grid_search.best_params_)
        return grid_search.best_params_


KNN = TitanicKNN
# preparing data (train & test)
train_data = KNN.data_preparing(file_name='train.csv')
train_data['Sex'] = KNN.encoder(data=train_data['Sex'])
x_train, y_train = KNN.data_separate(data=train_data)

test_data = KNN.data_preparing(file_name='test.csv')
test_data['Sex'] = KNN.encoder(data=test_data['Sex'])
x_test = test_data

best_parameters = KNN.grid_search(x_train, y_train)

# use model for prediction
result = KNN.model_predict(
        x_train, y_train,
        x_test,
        n_neighbors=best_parameters['n_neighbors']
)
test_data['Survived predict'] = result

# we guess, that all woman survived and all man died
submission_data = read_csv(
        'gender_submission.csv',
        sep=",", header=0
)

test_data['Survived true'] = submission_data['Survived']

cnt = 0
for index, row in test_data.iterrows():
    if row['Survived predict'] == row['Survived true']:
        cnt += 1

# compute accuracy
n_row = test_data.shape[0]
accuracy = cnt / n_row

print(cnt, n_row)
print('accuracy:', accuracy)
