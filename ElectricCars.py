#Uses Electric Cars dataset, shown here:
#https://www.kaggle.com/kkhandekar/quickest-electric-cars-ev-database

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

pd.set_option("display.max_rows", None, "display.max_columns", None)
data_set=pd.read_csv('C:/Cars/Quickestelectriccars-EVDatabase.csv')
data_set= data_set[data_set.FastChargeSpeed != '-']
#data_set= data_set[data_set.PriceinGermany != 'NaN']
data_set['PriceinGermany'].replace('', np.nan, inplace=True)
data_set= data_set[data_set.PriceinGermany != 'NaN']
#data_set= data_set[data_set.PriceinGermany != ' ']

data_set.dropna(subset=['PriceinGermany'])


#Drop name and subtitle, not important for modeling
#data_set.drop(['Name'],['Subtitle'])
data_set['Acceleration'] = data_set['Acceleration'].map(lambda x: x.lstrip('+-').rstrip(' sec'))
data_set['TopSpeed'] = data_set['TopSpeed'].map(lambda x: x.lstrip('+-').rstrip(' km/h'))
data_set['Range'] = data_set['Range'].map(lambda x: x.lstrip('+-').rstrip(' km'))
data_set['Efficiency'] = data_set['Efficiency'].map(lambda x: x.lstrip('+-').rstrip(' Wh/km'))
data_set['FastChargeSpeed'] = data_set['FastChargeSpeed'].map(lambda x: x.lstrip('+-').rstrip(' km/h'))
data_set["Acceleration"]=data_set.Acceleration.astype(float)
data_set["TopSpeed"]=data_set.TopSpeed.astype(float)
data_set["Range"]=data_set.Range.astype(float)
data_set["Efficiency"]=data_set.Efficiency.astype(float)
#data_set["FastChargeSpeed"]=data_set.FastChargeSpeed.astype(float)
# Can't convert to float for some reason
data_set['PriceinGermany'] = data_set['PriceinGermany'].astype(str).str.replace('\D+', '')


data_set["PriceinGermany"]=data_set.PriceinGermany.astype(float) # Gives an error, can't convert object to float

new_data=data_set[["Acceleration","TopSpeed","Range","Efficiency","FastChargeSpeed","PriceinGermany"]] #form new data_set
#Visualize data
x=new_data["Acceleration"]
y=new_data["TopSpeed"]
plt.scatter(x, y) #could be modeled via Linear reg.
z=new_data["Range"]
plt.scatter(x, z)
q=new_data["Efficiency"]
plt.scatter(x, q)
p=new_data["FastChargeSpeed"]
plt.scatter(x, p) #could be modeled via Linear reg.
model = sm.OLS(y, x).fit()
model.summary()


