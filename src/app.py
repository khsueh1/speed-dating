import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score, average_precision_score, precision_score
import warnings
import math
warnings.filterwarnings('ignore')
'''
Authors: Robert Xu, Kenny Hsueh
Date: 9/29/18
'''

#Initial Data Load
# Split data into training, validation, testing data
# Split data into 0.5% training, 0.25% validation, 0.25% testing
data = pd.read_csv(r'../Data/Speed Dating Data_Original.csv', encoding='latin1')
data = data[pd.notnull(data['int_corr'])]
training_data, init_testing_data = model_selection.train_test_split(data, test_size=0.5,random_state=42)
validation_data, testing_data = model_selection.train_test_split(init_testing_data, test_size=0.5, random_state=42)

print('Training Data Shape ',training_data.shape)
print('Validation Data Shape ',validation_data.shape)
print('Testing Data Shape ',testing_data.shape)

# Measure Performance
def measure_performance(true, pred, show_accuracy=True, show_f1=True, show_class_report=True, show_confusion_matrix=True):
    if show_f1:
        print("F1 score: {0: .3f}".format(f1_score(true, pred, average='macro')))
        print("Precision Score: {0: .3f}".format(precision_score(true, pred, average="macro")))
        print("Recall Score: {0: .3f}".format(recall_score(true, pred, average="macro")))
    if show_accuracy:
        print("Accuracy: {0: .3f}".format(accuracy_score(true, pred)))
    if show_class_report:
        print("Classification report: ")
        print(classification_report(true, pred),'\n')
    if show_confusion_matrix:
        print("Confusion matrix")
        print(confusion_matrix(true, pred),"\n")

#Prepare some training data
training_columns = ['samerace','int_corr','gender']
key_columns = ['match']
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(training_data[training_columns],training_data[key_columns],test_size=0.2,random_state=42)
# print(train_Y.head)
# print(train_X.head)
#Prepare some models
models = [('NB_Bernoulli', BernoulliNB()),
          ('LR', LogisticRegression(penalty='l2',max_iter=100,C=1.0)),
          ('LINEAR SVM',SVC(C=1000.0, cache_size=200,class_weight='balanced',coef0=0, degree=3, gamma='auto',kernel='linear',
                            max_iter=-1,probability=False,random_state=None,shrinking=True, tol=0.001, verbose=False)),
          ('RANDOM FOREST', RandomForestClassifier(n_estimators=200, criterion='gini',
                                                   max_depth=None, min_samples_split=2,
                                                   min_samples_leaf=1, max_features='auto',bootstrap=True,oob_score=False,
                                                   n_jobs=1, random_state=None, verbose=0))]
print('--- Training Models ---')
for model in models:
    clf = model[1]
    clf.fit(train_X, train_Y)
    y_predValid = clf.predict(test_X)
    print("{}:".format(model[0]))
    measure_performance(test_Y, y_predValid)
    print("------------------------------------------")


'''
I like the way you've set up the constants and fields and stuff
'''
""" Splits the provided data into train, cross validation, and test sets.
 Remaining percentage not included by trainPercentage and testPercentage is used for cross validation """
def splitData(data, trainPercentage, testPercentage):
    leftoverData, trainData = model_selection.train_test_split(data, test_size=trainPercentage, random_state=42)
    testCount = len(data) * testPercentage
    testScaledPercentage = testCount / len(leftoverData)
    testData, crossValidationData  = model_selection.train_test_split(leftoverData,
                                                                      test_size=testScaledPercentage,
                                                                      random_state=42)
    return trainData, crossValidationData, testData


# Constants
trainPercent = .6
crossValidationPercent = .2
testPercent = .2

# Initial Data Load
data = pd.read_csv('../Data/Speed Dating Data_Original.csv', encoding="ISO-8859-1", thousands=',')

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

# Split data into training, validation, and testing sets
match = data[data.match == 1]
noMatch = data[data.match == 0]

trainMatchData, crossValidationMatchData, testMatchData = splitData(match, testPercent, testPercent)
trainNoMatchData, crossValidationNoMatchData, testNoMatchData = splitData(noMatch, testPercent, testPercent)

trainData = trainMatchData + trainNoMatchData
crossValidationData = crossValidationMatchData + crossValidationNoMatchData
testData = testMatchData + testNoMatchData

# Train


# Cross Validation

# Test
