import numpy as np
import pandas as pd
import math
from sklearn import model_selection
import random

'''
Authors: Robert Xu, Kenny Hsueh
Date: 9/29/18
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
shuffleSeed = 4

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

trainData = random.Random(shuffleSeed).shuffle(trainMatchData + trainNoMatchData)
crossValidationData = random.Random(shuffleSeed).shuffle(crossValidationMatchData + crossValidationNoMatchData)
testData = random.Random(shuffleSeed).shuffle(testMatchData + testNoMatchData)

# Train


# Cross Validation

# Test
