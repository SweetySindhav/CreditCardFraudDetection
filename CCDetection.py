# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:13:02 2019

@author: sweetysindhav
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import scikit 

#print("PYTHON {}".format(pd.__version__))

data = pd.read_csv('creditcard.csv')
print(data.columns)
print(data.shape)
print(data.describe)
#data.shape and data.describe gives same o/p i.e 284807 x 31 so no missing values
print(data.describe())
#describe and describe()

data = data.sample(frac = 0.1 ,random_state = 1)
print(data.shape)

#plotting histogram
data.hist(figsize=(20,20))
plt.show()


#Indentifying fraud cases
fraud = data[data['Class']==1]
valid = data[data['Class']==0]
print("Fraud Cases: {} ".format(len(fraud)))
print("Valid Cases: {}".format(len(valid)))

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)

#Correlation Matrix

corrmat = data.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax = 0.8,square = True)
plt.show()

#Get the columns necessary ,here 'Class' removed

columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Class']]

#Store the variable we will be predicting
target = "Class"
X = data[columns]
Y = data[target]

#print shape of X and y
print(X,X.shape)
print(Y,Y.shape)

####################################################
#Applying Algorithm

from sklearn.metrics import classfication_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#can also be performed by svm but here we have large dataset ,
#so calculating support vector will take long


# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}
# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    # Run classification metrics
    print('{}\n Number oferrors: {}'.format(clf_name, n_errors))
    print("Accuracy: ",accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))