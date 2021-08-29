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

lrmodel=LogisticRegression(max_iter=1000) #increase iterations
train.dropna(inplace=True)
train=pd.get_dummies(train)
X=train.drop(['Survived'],axis=1)
Y=train['Survived']
#convert strings to floats, as part of model input (model can't accept strings)

lrmodel.fit(X,Y) # Run the model
y_hat=lrmodel.predict(test_X) # Create arracy of predicted Y values
#Check accuracy score
acc_log = round(lrmodel.score(X, Y) * 100, 2)
#Gives 79.21
#Note: Can't compare predicted Y against test Y ('Survived') values because the test.csv file doesn't include those values

#Can we do better than the calculated accuracy?
#Try the following: Remove 'PassengerID' from X data set as it is not needed for model
X = X.drop(['PassengerId'], axis=1)
# Create a new variable called AgeGroup, for age groups. There are 6 different age groups, from ages 0-11, 12-20, 21-30, 31-40, 41-50, and 51+
X['Age']=X['Age'].astype(int)
X.loc[X['Age'] <= 11, 'Age'] = 0
X.loc[(X['Age'] > 11) & (X['Age'] <= 20), 'Age'] = 1
X.loc[(X['Age'] > 20) & (X['Age'] <= 30), 'Age'] = 2

X.loc[(X['Age'] > 30) & (X['Age'] <= 40), 'Age'] = 3

X.loc[(X['Age'] > 40) & (X['Age'] <= 50), 'Age'] = 4

X.loc[(X['Age'] > 50), 'Age'] = 5

X['Fare']=X['Fare'].astype(int)
X.loc[X['Fare'] <= 8, 'Fare'] = 0
X.loc[(X['Fare'] > 8) & (X['Fare'] <= 15), 'Fare'] = 1

X.loc[(X['Fare'] > 15) & (X['Fare'] <= 30), 'Fare'] = 2


X.loc[(X['Fare'] > 30) & (X['Fare'] <= 60), 'Fare'] = 3


X.loc[(X['Fare'] > 60) & (X['Fare'] <= 175), 'Fare'] = 4

X.loc[(X['Fare'] > 175), 'Fare'] = 5

# Re-run model
lrmodel.fit(X,Y) # Run the model
#Check accuracy score
acc_log = round(lrmodel.score(X, Y) * 100, 2)
#Now 79.78

#Try Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, Y)

acc_random_forest = round(random_forest.score(X, Y) * 100, 2)
# This gives accuracy score of 89.61
