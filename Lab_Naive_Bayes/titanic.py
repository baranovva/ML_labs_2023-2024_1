from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from numpy.random import seed

seed(2023)


class TitanicGaussianNB:
    def data_preparing(file_name):  # filling missing data and del some data classes
        data = read_csv(file_name)

        mid_age = data['Age'].median()
        data['Age'] = data['Age'].fillna(mid_age)
        data['Relatives'] = data['SibSp'] + data['Parch']

        data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare',
                                  'Cabin', 'Embarked', 'SibSp', 'Parch'])
        return data

    def encoder(data):  # create labels for categorical data
        coder = LabelEncoder()
        coder.fit_transform(data)
        return coder.transform(data)

    def data_separate(data):  # separate data as input and output
        x = data[['Pclass', 'Sex', 'Age', 'Relatives']]
        y = data['Survived']
        return x, y

    def model_predict(x_train, y_train, x_test):  # fit model and take predict
        model = GaussianNB()
        model.fit(x_train, y_train)
        return model.predict(x_test)


# preparing data (train & test)
train_data = TitanicGaussianNB.data_preparing(file_name='train.csv')
train_data['Sex'] = TitanicGaussianNB.encoder(data=train_data['Sex'])
x_train, y_train = TitanicGaussianNB.data_separate(data=train_data)

test_data = TitanicGaussianNB.data_preparing(file_name='test.csv')
test_data['Sex'] = TitanicGaussianNB.encoder(data=test_data['Sex'])
x_test = test_data

# use model for prediction
result = TitanicGaussianNB.model_predict(x_train, y_train, x_test)
test_data['Survived predict'] = result

# we guess, that all woman survived and all man died
submission_data = read_csv('gender_submission.csv', sep=",", header=0)
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
