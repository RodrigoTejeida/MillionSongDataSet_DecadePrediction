import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC

# Import Data Set Using Pandas
data = pd.read_csv('/Users/rodrigotejeida/Desktop/Big Data Project/YearPredictionMSD.txt', header = None)

# Make train/test split
train = data[0:463715]
test  = data[463715:515345]
len(data)==len(train)+len(test)

# Define the independent and dependent variables
x = train[train.columns[1:91]]
y = train[train.columns[0:1 ]]

# Preliminary Data Analysis

y = np.round(y/10)
y = pd.DataFrame(y)

#  Songs by Year
px = np.sort(y[0].unique())
py = y[0].groupby(y[0]).count()
plt.ylabel('Number of Songs')
plt.xlabel('Decade')
plt.title('Songs by Decade')
plt.bar(px, py, align='center')

# Avg Timbre by year
ts    = train.sort_values([0])
px2   = ts[ts.columns[0 ]].unique()
py21  = ts[ts.columns[1 ]].groupby(ts[0]).mean()
py22  = ts[ts.columns[2 ]].groupby(ts[0]).mean()
py23  = ts[ts.columns[3 ]].groupby(ts[0]).mean()
py24  = ts[ts.columns[4 ]].groupby(ts[0]).mean()
py25  = ts[ts.columns[5 ]].groupby(ts[0]).mean()
py26  = ts[ts.columns[6 ]].groupby(ts[0]).mean()
py27  = ts[ts.columns[7 ]].groupby(ts[0]).mean()
py28  = ts[ts.columns[8 ]].groupby(ts[0]).mean()
py29  = ts[ts.columns[9 ]].groupby(ts[0]).mean()
py210 = ts[ts.columns[10]].groupby(ts[0]).mean()
py211 = ts[ts.columns[11]].groupby(ts[0]).mean()
py212 = ts[ts.columns[12]].groupby(ts[0]).mean()

plt.ylabel('Average of Timbre group for given year')
plt.xlabel('Year')
plt.title('Timbre by Year')
plt.scatter(px2, py21,  s=1)
plt.scatter(px2, py22,  s=1)
plt.scatter(px2, py23,  s=1)
plt.scatter(px2, py24,  s=1)
plt.scatter(px2, py25,  s=1)
plt.scatter(px2, py26,  s=1)
plt.scatter(px2, py27,  s=1)
plt.scatter(px2, py28,  s=1)
plt.scatter(px2, py29,  s=1)
plt.scatter(px2, py210, s=1)
plt.scatter(px2, py211, s=1)
plt.scatter(px2, py212, s=1)

Xv = list()
Xv.append([x[x.columns[0]]])
Xv.append([x[x.columns[1]]])
Xv.append([x[x.columns[2]]])
Xv.append([x[x.columns[3]]])
Xv.append([x[x.columns[4]]])
Xv.append([x[x.columns[5]]])
Xv.append([x[x.columns[6]]])
Xv.append([x[x.columns[7]]])
Xv.append([x[x.columns[8]]])
Xv.append([x[x.columns[9]]])
Xv.append([x[x.columns[10]]])
Xv.append([x[x.columns[11]]])

pos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.violinplot(Xv, pos, points=40, widths=0.5, showmeans=True,
                      showextrema=True, showmedians=True, bw_method='silverman')
plt.title('Violin plot of Timbre Average', fontsize=10)

y.describe()
y.mode()





# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
#
# axes[0, 1].violinplot(x[1:], pos, points=40, widths=0.5,
#                       showmeans=True, showextrema=True, showmedians=True,
#                       bw_method='silverman')
# axes[0, 1].set_title('Violin plot of Timbre average', fontsize=10)
#
#
# d = [np.random.normal(0, std, size=100) for std in pos]



t1  = [1.89,4.26]
t2  = [55/60,1.5]
tt1 = [.9,1.9]
tt2 = [1.1,2.1]

plt.ylabel('Time')
plt.title('Pythons vs Spark')
plt.bar(tt1, t1, width=0.3,color='c',align='center')
plt.bar(tt2,t2, width=0.3,color='gray',align='center')
plt.legend(('Python','Spark'))
i = np.arange(2)+.85
plt.xticks(i + .3/2,('1 Epoch','2 Epochs'))






























