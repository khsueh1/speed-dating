import numpy as np
import pandas as pd
from sklearn import model_selection

'''
Authors: Robert Xu, Kenny Hsueh
Date: 9/29/18
'''


#Initial Data Load
# Split data into training, validation, testing data
# Split data into x% training, x% validation, x% testing



data = pd.read_csv('../Data/Speed Dating Data_Original.csv', encoding = "ISO-8859-1")
trainingdata, testing_data = model_selection.train_test_split(data, test_size=0.5, random_state=42)

