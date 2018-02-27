import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

# Import Data Set Using Pandas
data = pd.read_csv('/Users/rodrigotejeida/Desktop/Big Data Project/YearPredictionMSD.txt', header = None)

# Make train/test split
train = data[0:463715]
test  = data[463715:515345]
len(data)==len(train)+len(test)

x  = train[train.columns[1:91]]
y  = train[train.columns[0:1 ]]-1923
xt = test[test.columns[1:91]]
yt = test[test.columns[0:1 ]]-1923

# Linear Regression
regr   = linear_model.LinearRegression()
regr.fit(x, y)
yh_lr  = np.round(regr.predict(xt))
MAE_lr = np.sum(np.abs(yt-yh_lr))[0]/yt.size
MAE_lr

# Ridge Regression
rdg = Ridge(alpha=1000)
rdg.fit(x,y)
yh_rdg = np.round(rdg.predict(xt))
MAE_rdg = np.sum(np.abs(yt-yh_rdg))[0]/yt.size
MAE_rdg

# Lasso
lass = linear_model.Lasso(alpha=0.1)
lass.fit(x,y)
yh_lass = np.round(lass.predict(xt))
yh_lass = yh_lass.reshape(yh_lass.size,1)
MAE_lass = np.sum(np.abs(yt-yh_lass))[0]/yt.size
MAE_lass

mx  = x.values
mxt = xt.values
my  = y.values
myt = yt.values
my = my.reshape(my.shape[0])

# Random Forrest
rf = RandomForestClassifier(n_estimators=15, max_depth=10, random_state=0)
rf.fit(mx,my)
yh_rf = np.round(rf.predict(mxt))
yh_rf = yh_rf.reshape(yh_rf.size,1)
MAE_rf = np.sum(np.abs(myt-yh_rf))/myt.size
MAE_rf

x  = x.values
xt = xt.values
y  = y.values
yt = yt.values

y = y[:,0]


















