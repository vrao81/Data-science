#Using Iris Data Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
iris_data = load_iris()
X= iris_data.data
Y=iris_data.target

#First try logistic regression,
X_data=pd.DataFrame(data=X, columns=["sepal_l","sepal_w","petal_l","petal_w"]) #create new dataframe with data
X_data["class"]=Y
X_shuffled = X_data.sample(frac=1).reset_index(drop=True) #shuffle the data
# Select training and test data
X_train=X_shuffled[0:75]
X_test=X_shuffled[75:]
# This splits the dataframe into 2. The train data set is used to train the model, test is to evaluate it
Y_train=X_train["class"]
X_train.drop("class", axis=1)
Y_test=X_test["class"]
X_test.drop("class", axis=1)
#Alternately can use test_train_slit from sklearn
#Now run the model
lrmodel=LogisticRegression(max_iter=1000)
lrmodel.fit(X_train,Y_train)
y_predicted=lrmodel.predict(X_test) # Create predicted values

#Now compare predicted values (y_predicted, versus Y_test)
print(classification_report(Y_test, y_predicted))
#Model correctly classified every species

#Now try decision classifier
clf = DecisionTreeClassifier(random_state=0)


clf.fit(X_train, Y_train)
y_predicted=clf.predict(X_test)
print(classification_report(Y_test, y_predicted)) #Again 100% accuracy
print(clf.get_depth()) # Get depth of tree
print(clf.get_n_leaves())
#If change DT parameters, 
clf = DecisionTreeClassifier(splitter = 'best', max_leaf_nodes=2, random_state=0)

clf.fit(X_train, Y_train)
y_predicted=clf.predict(X_test)
print(classification_report(Y_test, y_predicted)) #Accuracy goes down