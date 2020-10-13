#Logistic regression model for titanic dataset, with 91% accuracy in predicting
survival


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

pd.set_option("display.max_rows", None, "display.max_columns", None)
train=pd.read_csv('C:/Titanic/train.csv')
test=pd.read_csv('C:/Titanic/test.csv')
#drop unecessary columns
train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)



lrmodel=LogisticRegression(max_iter=170)
#create training data X and Y variables
train.dropna(inplace=True)
train=pd.get_dummies(train)


X=train.drop(['Survived'],axis=1)
Y=train['Survived']
#convert strings to floats, as part of model input (model can't accept strings)




lrmodel.fit(X,Y)
#now test model fit
test.dropna(inplace=True)
test=pd.get_dummies(test)
test_X=test.drop(['Survived'],axis=1)
test_Y=test['Survived']


y_hat=lrmodel.predict(test_X)
conf = confusion_matrix(y_hat,test_Y)
sns.heatmap(conf,annot=True,cmap=plt.cm.plasma)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()
print(accuracy_score(y_hat,test_Y))
#returns 91% accuracy. Can we do better?

#Drop additional columns?


train.drop(['SibSp','Parch'],axis=1,inplace=True)
test.drop(['SibSp','Parch'],axis=1,inplace=True)
X=train.drop(['Survived'],axis=1)
Y=train['Survived']

lrmodel.fit(X,Y)

test.dropna(inplace=True)
test=pd.get_dummies(test)
test_X=test.drop(['Survived'],axis=1)
test_Y=test['Survived']
y_hat=lrmodel.predict(test_X)
conf = confusion_matrix(y_hat,test_Y)
sns.heatmap(conf,annot=True,cmap=plt.cm.plasma)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()
print(accuracy_score(y_hat,test_Y))
# results in 90% accuracy
