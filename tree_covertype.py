#Data Analysis using the tree cover type dataset, part of sklearn
#Uses XGBoost, a gradient boosting algorithm optimized for speed and performance compared to random forests or decision trees. For more information see: https://xgboost.readthedocs.io/en/latest/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb

data_set=fetch_covtype()
# See reference material here for information on cover type classes: https://archive.ics.uci.edu/ml/datasets/Covertype
#Create labels for classes from above website
Cover_Types = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
column_list=['elevation', 'aspect', 'slope', 'hor_dist_hydro', 'vert_dist_hydro', 'hor_dist_road', 'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
'hor_dist_fire']+[f"wilderness_{i}" for i in range(4)]+[f"soil_type_{i}" for i in range(40)]

df=pd.DataFrame(data_set.data, columns=column_list)
df['labels']=data_set.target

#create test and training data
train,test=train_test_split(df, test_size=0.2, random_state=32)
train=train.sample(10000) #random sample

X_data=pd.DataFrame(train.groupby('labels').size(), columns=['labels'])
X_data.index=X_data.index.map(Cover_Types)

plt.figure(figsize=(12,12))
sns.barplot(data=X_data.T)

fig, ax=plt.subplots(figsize=(16,16))
pd.plotting.scatter_matrix(train.loc[:, [col for col in train.columns if ("soil_type" not in col) and ("wilderness" not in col)]].sample(500), ax=ax)

#Now ready to run algorithm
X=train.drop('labels', axis=1)
Y=train['labels'] #Create X and Y training data

clf=xgb.XGBClassifier(n_jobs=2)
clf.fit(X,Y)

yhat=clf.predict(X)
f1_score(Y, yhat, average='weighted')
#gives #0.95 score

X_test=test.drop('labels', axis=1)
Y_test=test['labels']

yhat_test=clf.predict(X_test)
f1_score(Y_test, yhat_test, average='weighted')
#Gives 0.7987

#So our score went down - due to Overfitting
#Change model parameters to examine how it impacts model prediction

clf2 = xgb.XGBClassifier(n_jobs=2, max_depth=13, n_estimators=500)
 # Choose a larger tree depth and more trees for a more complex model
clf2.fit(X,Y)
yhat=clf2.predict(X)
f1_score(Y, yhat, average='weighted')

#Gives score of 1.0, so perfect fit to training data

yhat_test=clf2.predict(X_test)
f1_score(Y_test, yhat_test, average='weighted')
#Gives 0.824, so better than before

#Now try max_depth=1,  for illustration
clf3 = xgb.XGBClassifier(n_jobs=2, max_depth=1, n_estimators=500)
clf3.fit(X,Y)
yhat=clf3.predict(X)
f1_score(Y, yhat, average='weighted')
#gives 0.764, so went down due to less complex model

yhat_test=clf3.predict(X_test)
f1_score(Y_test, yhat_test, average='weighted')
#Gives 0.724


