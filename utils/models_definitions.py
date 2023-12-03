
######################################## IMPORTS ########################################

import numpy as np
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb


################################## TRAIN AND TEST SETS ##################################

training_set = pd.read_csv('../dataset/standardized_training.csv')
X_train = training_set.loc[:,:'five_year']
y_train = training_set['general_two_year'].values

test_set = pd.read_csv('../dataset/standardized_testing.csv')
X_test = test_set.loc[:,:'five_year']
y_test = test_set['general_two_year'].values


################################### MODEL DEFINITIONS ###################################