# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 10:46:42 2021

@author: joanne
"""

import csv
from re import split
import numpy as np
import csv
from numpy.core.numeric import NaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

np.random.seed(123)


df = pd.read_csv(r'C:\Users\user001\OneDrive\桌面\NTU\senior\Machine_Learning\final_project\HTML_final_project-main\trainval_data.csv')
df = df.iloc[0:,0:46]

##
#encoding starts
objList = df.select_dtypes(include = "object").columns
#print (objList)

le = LabelEncoder()

for feat in objList:
    df[feat] = le.fit_transform(df[feat].astype(str))


path = 'output_data.csv'
np.savetxt(path, df, delimiter =',', fmt ='%s')
