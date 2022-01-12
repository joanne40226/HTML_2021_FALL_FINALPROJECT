# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from numpy import linalg
from IPython import get_ipython
get_ipython().magic('reset -sf')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import csv

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def make_pred(pred, test_id):
    with open('pred.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i in range(1407):
            writer.writerow([test_id[i],pred[i]])

def data_load(path_,):
    df= pd.read_csv(path_)
    dy = df['Churn_Category']
    dx = df.drop(['Churn_Category'], axis=1)
    return dx,dy

def data_process(d):
    df = d.drop(['Customer ID','Count','Under 30','Senior Citizen','Dependents','Country','State','City','Lat Long','Latitude','Longitude','Quarter','Referred a Friend'], axis=1)    

    df = df.replace('Offer A',1)
    df = df.replace('Offer B',2)
    df = df.replace('Offer C',3)
    df = df.replace('Offer D',4)
    df = df.replace('Offer E',5)
    df = df.replace('Yes',1)
    df = df.replace('No',0)
    df = df.replace('None',0)
    df = df.replace('Female',1)    
    df = df.replace('Male',2)
    df = df.replace({'Gender': np.nan},1.507892293)
    df = df.replace({'Age' : np.nan},46.78522868)
    df = df.replace({'Married': np.nan},0.466172382)
    df = df.replace({'Number of Dependents': np.nan},0.458721291)
    df = df.replace({'Zip Code':np.nan},93460.99382)
    df = df.replace({'Satisfication Score':np.nan},3.260452152)
    df = df.replace({'Number of Referrals': np.nan},1.840825123)
    df = df.replace({'Tenure in Months':np.nan},32.40104487)
    df = df.replace({'Offer': np.nan},1.390003085)
    df = df.replace({'Phone Service':np.nan},0.905519177)
    df = df.replace({'Avg Monthly Long Distance Charges':np.nan},23.00672558)
    df = df.replace({'Multiple Lines': np.nan},0.417030568)
    df = df.replace({'Internet Service': np.nan},0.774630542)
    df = df.replace({'Internet Type': np.nan},1.370243294)
    df = df.replace('Fiber Optic',1)
    df = df.replace('Cable',2)
    df = df.replace('DSL',3)
    df = df.replace({'Internet Type': np.nan},1.370243294)
    df = df.replace({'Internet Type': np.nan},1.370243294)
    df = df.replace({'Internet Type': np.nan},1.370243294)
    df = df.replace({'Internet Type': np.nan},1.370243294)
    df = df.replace({'Avg Monthly GB Download': np.nan},20.32366698)
    df = df.replace({'Online Security': np.nan},0.284034653)
    df = df.replace({'Online Backup': np.nan},0.334686232)
    df = df.replace({'Device Protection Plan': np.nan},0.333333333)
    df = df.replace({'Premium Tech Support': np.nan},0.284119107)
    df = df.replace({'Streaming TV':np.nan},0.377661216)
    df = df.replace({'Streaming Movies':np.nan},0.379523662)
    df = df.replace({'Streaming Music':np.nan},0.346009975)
    df = df.replace({'Unlimited Data': np.nan},0.666460778)
    df = df.replace({'Contract': np.nan},292.0086393)
    df = df.replace('One Year',365)
    df = df.replace('Two Year',730)
    df = df.replace('Month-to-Month',30)
    df = df.replace({'Paperless Billing': np.nan},0.578473945)
    df = df.replace({'Payment Method': np.nan},1.505276226)
    df = df.replace('Bank Withdrawal',1)
    df = df.replace('Credit Card',2)
    df = df.replace('Mailed Check',3)
    df = df.replace({'Monthly Charge': np.nan},64.19761171)
    df = df.replace({'Total Charges': np.nan},2259.597129)
    df = df.replace({'Total Refunds': np.nan},2.040018645)
    df = df.replace({'Total Extra Data Charges':np.nan},6.506550218)
    df = df.replace({'Total Long Distance Charges':np.nan},751.9429664)
    df = df.replace({'Total Long Distance Charges': np.nan},751.9429664)
    df = df.replace({'Total Long Distance Charges': np.nan},751.9429664)
    df = df.replace({'Total Long Distance Charges': np.nan},751.9429664)    
    df = df.replace({'Total Revenue':np.nan},3025.38111)
    df = df.replace('No Churn',0)
    df = df.replace('Competitor',1)
    df = df.replace('Dissatisfaction',2)
    df = df.replace('Attitude',3)
    df = df.replace('Price',4)
    df = df.replace('Other',5)
    return df



    
def PCA_SV(xtrain,ytrain,xtest,ytest):
    steps = [('scaler', StandardScaler()),('pca',PCA()),('clf',SVC(kernel='rbf'))]
    parameters = {
        'pca__n_components' :[2,3,4],
        'clf__C':[0.001,0.1,0.01,1,10,100,10e5],
        'clf__gamma':[1,0.1,0.01,0.001]
    }
    pipeline = Pipeline(steps)
    
    cv=5
    grid = GridSearchCV(pipeline,param_grid=parameters,cv=cv)
    grid.fit(xtrain,ytrain)
    print("Score for %d fold : = %f"%(cv,grid.score(xtest,ytest)))
    print("Parameters : ",grid.best_params_)
    y_pred_test = grid.predict(xtest)
    print("Accuracy : ", grid.best_score_)
    return y_pred_test

path = r'C:\Users\user001\OneDrive\桌面\NTU\senior\Machine_Learning\final_project\HTML_final_project-main\TrainID_NaN.csv'
#dff= pd.read_csv(path)
dff= pd.read_csv(path)
df = data_process(dff)
#dx = df.drop['Churn_Category']
X = df.drop(['Churn_Category'], axis=1)
y = df['Churn_Category']
#dx = data_init_load(path)[0]
#dy = data_init_load(path)[1]
xtrain, xtest,  ytrain, ytest = train_test_split(X, y, test_size=0.3)
#test_prediction = PCA_SV(X_train,y_train,X_test,y_test)

steps = [('scaler', StandardScaler()),('pca',PCA()),('clf',SVC(kernel='rbf'))]
parameters = {
    'pca__n_components' :[2,3,4],
    'clf__C':[0.001,0.1,0.01,1,10,100,10e5],
    'clf__gamma':[1,0.1,0.01,0.001]
}
pipeline = Pipeline(steps)

cv=5
grid = GridSearchCV(pipeline,param_grid=parameters,cv=cv)
grid.fit(xtrain,ytrain)
print("Score for %d fold : = %f"%(cv,grid.score(xtest,ytest)))
print("Parameters : ",grid.best_params_)
y_pred_test = grid.predict(xtest)
print("Accuracy : ", grid.best_score_)


path_test = r'C:\Users\user001\OneDrive\桌面\NTU\senior\Machine_Learning\final_project\HTML_final_project-main\Test_data.csv'
dff_t= pd.read_csv(path_test)
X_test = data_process(dff_t)
Xtest = X_test.drop(['Churn_Category'], axis=1)
test_id = np.array(Xtest[1:])
#test_prediction = PCA_SV(X_train,y_train,X_test,y_test)
y_pred_test_tt = grid.predict(Xtest)
print("Accuracy : ", grid.best_score_)

make_pred(y_pred_test_tt,test_id)



#test_prediction = PCA_SV(X_train,y_train,X_test,y_test)


