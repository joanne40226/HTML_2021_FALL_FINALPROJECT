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
    df = df.replace('Fiber Optic',1)
    df = df.replace('Cable',2)
    df = df.replace('DSL',3)
    df = df.replace('One Year',365)
    df = df.replace('Two Year',730)
    df = df.replace('Month-to-Month',30)
    df = df.replace('Bank Withdrawal',1)
    df = df.replace('Credit Card',2)
    df = df.replace('Mailed Check',3)
    df = df.replace('No Churn',0)
    df = df.replace('Competitor',1)
    df = df.replace('Dissatisfaction',2)
    df = df.replace('Attitude',3)
    df = df.replace('Price',4)
    df = df.replace('Other',5)
    Gender_m = df['Gender'].mean()
    df = df.replace({'Gender': np.nan},Gender_m)
    Age_m = df['Age'].mean()
    df = df.replace({'Age' : np.nan},Age_m)
    Married = df['Married'].mean()
    df = df.replace({'Married': np.nan},Married)
    Dependents = df['Number of Dependents'].mean()
    df = df.replace({'Number of Dependents': np.nan},Dependents)
    Zip = df['Zip Code'].mean()
    df = df.replace({'Zip Code':np.nan},Zip)
    sat = df['Satisfication Score'].mean()
    df = df.replace({'Satisfication Score':np.nan},sat)
    Referrals = df['Number of Referrals'].mean()
    df = df.replace({'Number of Referrals': np.nan},Referrals)
    Tenure = df['Tenure in Months'].mean()
    df = df.replace({'Tenure in Months':np.nan},Tenure)
    Offer = df['Offer'].mean()
    df = df.replace({'Offer': np.nan},Offer)
    Phone = df['Phone Service'].mean()
    df = df.replace({'Phone Service':np.nan},Phone)
    Distance_c = df['Avg Monthly Long Distance Charges'].mean()
    df = df.replace({'Avg Monthly Long Distance Charges':np.nan},Distance_c)
    Multiple = df['Multiple Lines'].mean()
    df = df.replace({'Multiple Lines': np.nan},Multiple)
    Internet_Service = df['Internet Service'].mean()
    df = df.replace({'Internet Service': np.nan},Internet_Service)
    Internet_Type = df['Internet Type'].mean()
    df = df.replace({'Internet Type': np.nan},Internet_Type)
    GBDownload = df['Avg Monthly GB Download'].mean()
    df = df.replace({'Avg Monthly GB Download': np.nan},GBDownload)
    Online_Security = df['Online Security'].mean()
    df = df.replace({'Online Security': np.nan},Online_Security)
    Online_Backup = df['Online Backup'].mean()
    df = df.replace({'Online Backup': np.nan},Online_Backup)
    Protection = df['Device Protection Plan'].mean()
    df = df.replace({'Device Protection Plan': np.nan},Protection)
    Premium = df['Premium Tech Support'].mean()
    df = df.replace({'Premium Tech Support': np.nan},Premium)
    TV = df['Streaming TV'].mean()
    df = df.replace({'Streaming TV':np.nan},TV)
    MOV = df['Streaming Movies'].mean()
    df = df.replace({'Streaming Movies':np.nan},MOV)
    music = df['Streaming Music'].mean()
    df = df.replace({'Streaming Music':np.nan},music)
    Unlimited_Data = df['Unlimited Data'].mean()
    df = df.replace({'Unlimited Data': np.nan},Unlimited_Data)
    Contract = df['Contract'].mean()
    df = df.replace({'Contract': np.nan},Contract)
    Paperless = df['Paperless Billing'].mean()
    df = df.replace({'Paperless Billing': np.nan},Paperless)
    Payment = df['Payment Method'].mean()
    df = df.replace({'Payment Method': np.nan},Payment)
    Monthly_Charge = df['Monthly Charge'].mean()
    df = df.replace({'Monthly Charge': np.nan},Monthly_Charge)
    Total_Charges = df['Total Charges'].mean()
    df = df.replace({'Total Charges': np.nan},Total_Charges)
    Total_Refunds = df['Total Refunds'].mean()
    df = df.replace({'Total Refunds': np.nan},Total_Refunds)
    Total_Extra_Data_Charges = df['Total Extra Data Charges'].mean()
    df = df.replace({'Total Extra Data Charges':np.nan},Total_Extra_Data_Charges)
    Total_Long_Distance_Charges = df['Total Long Distance Charges'].mean()
    df = df.replace({'Total Long Distance Charges':np.nan},Total_Long_Distance_Charges) 
    Total_Revenue = df['Total Revenue'].mean()  
    df = df.replace({'Total Revenue':np.nan},Total_Revenue)
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


