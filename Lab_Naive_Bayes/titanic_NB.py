from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from numpy.random import seed

seed(2023)


class TitanicGaussianNB:
    # filling missing data and del some data classes
    def data_preparing(file_name):
        data = read_csv(file_name)

        mid_age = data['Age'].median()
        data['Age'] = data['Age'].fillna(mid_age)
        data['Relatives'] = data['SibSp'] + data['Parch']

        data = data.drop(
                columns=[
                    'PassengerId', 'Name', 'Ticket', 'Fare',
                    'Cabin', 'Embarked', 'SibSp', 'Parch'
                ]
        )
        return data

    # create labels for categorical data
    def encoder(data):
        coder = LabelEncoder()
        coder.fit_transform(data)
        return coder.transform(data)

    # separate data as input and output
    def data_separate(data):
        x = data[['Pclass', 'Sex', 'Age', 'Relatives']]
        y = data['Survived']
        return x, y

    # fit model and take predict
    def model_predict(
            x_train, y_train,
            x_test, y_test
    ):
        model = GaussianNB()
        model.fit(x_train, y_train)
        return model.score(x_test, y_test)


NB = TitanicGaussianNB
# preparing data (train & test)
train_data = NB.data_preparing(file_name='train.csv')
train_data['Sex'] = NB.encoder(data=train_data['Sex'])
x_train, y_train = NB.data_separate(data=train_data)

test_data = NB.data_preparing(file_name='test.csv')
test_data['Sex'] = NB.encoder(data=test_data['Sex'])
x_test = test_data

submission_data = read_csv(
        'gender_submission.csv',
        sep=",", header=0
)
y_test = submission_data['Survived']

# use model for prediction
score = NB.model_predict(
        x_train, y_train,
        x_test, y_test
)
print(score)
