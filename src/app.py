import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score, \
    average_precision_score, precision_score
import warnings
import math

warnings.filterwarnings('ignore')

'''
Authors: Robert Xu, Kenny Hsueh
Date: 9/29/18
'''

# region Constants
TRAIN_PERCENTAGE = .6
CROSS_VALIDATION_PERCENTAGE = .2
TEST_PERCENTAGE = .2
RANDOM_STATE = 42


# endregion


def clean_data(data):
    """
    :param data:
    :return:
    """
    # TODO: This is temporary
    data = data[pd.notnull(data['int_corr'])]

    # region TODO: might not need any of this
    careers = {}
    fields = {}
    race = {
        1: 'Black/African',
        2: 'European/Caucasian-American',
        3: 'Latino/Hispanic',
        4: 'Asian/Pacific Islander/Asian-American',
        5: 'Native American',
        6: 'Other',
    }
    gender = {
        0: 'Female',
        1: 'Males',
    }

    for row in data.itertuples():
        if not math.isnan(row.field_cd):
            fields[row.field_cd] = row.field

        if not math.isnan(row.career_c):
            careers[row.career_c] = row.career

    print(race)
    print(gender)
    print(careers)
    print(fields)
    # endregion

    return data


def split_data_equally(data):
    """
    Splits data in train, cross validation, and test sets.
    Ensures an equal amount of positive and negative record in each data set.

    :param data:
    :return:
    """
    match = data[data.match == 1]
    no_match = data[data.match == 0]

    train_match_data, cross_validation_match_data, test_match_data = split_data(match, TRAIN_PERCENTAGE,
                                                                                TEST_PERCENTAGE)
    train_no_match_data, cross_validation_no_match_data, test_no_match_data = split_data(no_match, TRAIN_PERCENTAGE,
                                                                                         TEST_PERCENTAGE)

    # TODO: "randomly" shuffle the data?
    train_data = train_match_data + train_no_match_data
    cross_validation_data = cross_validation_match_data + cross_validation_no_match_data
    test_data = test_match_data + test_no_match_data

    return train_data, cross_validation_data, test_data


def split_data(data, train_percentage, test_percentage):
    """
    Splits the provided data into train, cross validation, and test sets.
    Remaining percentage not included by trainPercentage and testPercentage is used for cross validation.

    :param data: cleansed speed dating data set
    :param train_percentage: relative to 100
    :param test_percentage: relative to 100 (not relative to remaining percentage from train percentage)
    :return:
    """
    leftover_data, train_data = model_selection.train_test_split(data, test_size=train_percentage,
                                                                 random_state=RANDOM_STATE)
    test_count = len(data) * test_percentage
    test_scaled_percentage = test_count / len(leftover_data)
    test_data, cross_validation_data = model_selection.train_test_split(leftover_data,
                                                                        test_size=test_scaled_percentage,
                                                                        random_state=RANDOM_STATE)
    return train_data, cross_validation_data, test_data


# Initial Data Load
data = pd.read_csv(r'../Data/Speed Dating Data_Original.csv', encoding='latin1')

# Clean Data
data = clean_data(data)
print('data shape', data.shape)

# Split data into training, validation, testing
training_data, validation_data, testing_data = split_data_equally(data)
print('Training Data Shape ', training_data.shape)
print('Validation Data Shape ', validation_data.shape)
print('Testing Data Shape ', testing_data.shape)


# Measure Performance
def measure_performance(true, pred, show_accuracy=True, show_f1=True, show_class_report=True,
                        show_confusion_matrix=True):
    if show_f1:
        print("F1 score: {0: .3f}".format(f1_score(true, pred, average='macro')))
        print("Precision Score: {0: .3f}".format(precision_score(true, pred, average="macro")))
        print("Recall Score: {0: .3f}".format(recall_score(true, pred, average="macro")))
    if show_accuracy:
        print("Accuracy: {0: .3f}".format(accuracy_score(true, pred)))
    if show_class_report:
        print("Classification report: ")
        print(classification_report(true, pred), '\n')
    if show_confusion_matrix:
        print("Confusion matrix")
        print(confusion_matrix(true, pred), "\n")

    return


# Prepare some training data
training_columns = ['samerace', 'int_corr', 'gender']
key_columns = ['match']
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(training_data[training_columns],
                                                                    training_data[key_columns], test_size=0.2,
                                                                    random_state=RANDOM_STATE)
# print(train_Y.head)
# print(train_X.head)
# Prepare some models
models = [('NB_Bernoulli', BernoulliNB()),
          ('LR', LogisticRegression(penalty='l2', max_iter=100, C=1.0)),
          ('LINEAR SVM',
           SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0, degree=3, gamma='auto', kernel='linear',
               max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)),
          ('RANDOM FOREST', RandomForestClassifier(n_estimators=200, criterion='gini',
                                                   max_depth=None, min_samples_split=2,
                                                   min_samples_leaf=1, max_features='auto', bootstrap=True,
                                                   oob_score=False,
                                                   n_jobs=1, random_state=None, verbose=0))]

print('--- Training Models ---')
for model in models:
    clf = model[1]
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    clf.fit(train_X, train_Y)
    y_predValid = clf.predict(test_X)
    print("{}:".format(model[0]))
    measure_performance(test_Y, y_predValid)
    print("------------------------------------------")
