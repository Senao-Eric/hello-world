from sklearn import svm, grid_search, preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import cross_validation
from sklearn.utils import check_arrays
import csv
import pandas as pd
import numpy as np
import math
import time
import itertools
from pandas import Series, DataFrame

#train = pd.read_csv("D:/SPMD-DAS1/triptime_features_all.csv")
#label = pd.read_csv("D:/SPMD-DAS1/triptime_with_brake_in_city_all.csv")
features_train = pd.read_csv("D:/Taxi trip time/in_city/Sample_section_90_80_ignore_features.csv")
label_train = pd.read_csv("D:/Taxi trip time/in_city/Sample_section_90_80_ignore_triptime.csv")
features_test = pd.read_csv("D:/Taxi trip time/in_city/testing_2500_in_city_no_outlier_features.csv")
label_test = pd.read_csv("D:/Taxi trip time/in_city/testing_2500_in_city_no_outlier_triptime.csv")

train_array = features_train.iloc[:,0:].values
label_train_array = label_train.iloc[:,0:].values
test_array = features_test.iloc[:,0:].values
label_test_array = label_test.iloc[:,0:].values

label_train_final = list(itertools.chain.from_iterable(label_train_array))
label_train_final = np.asarray(label_train_final)
label_test_final = list(itertools.chain.from_iterable(label_test_array))
label_test_final = np.asarray(label_test_final)

#Normalization
scaler = preprocessing.StandardScaler().fit(train_array)
train_array = scaler.transform(train_array)
scaler = preprocessing.StandardScaler().fit(test_array)
test_array = scaler.transform(test_array)
#normalizer = preprocessing.Normalizer().fit(trainarray)
#trainarray = normalizer.transform(trainarray)

#print trainarray_scaled
#¦Û°Ê¤À³Îtraining¤Îtesting set
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainarray, labelfinal, test_size=0.2, random_state =0)
#c_range = np.logspace(0, 6, 10)
#param_grid=dict(C=c_range
#parameters = {'C':[1, 10, 100, 1000, 10000, 100000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'epsilon':[0.1, 0.01, 0.001, 0.0001, 0.00001]}
clf = svm.SVR(kernel='rbf', C=1000, gamma=0.1)
#clf = grid_search.GridSearchCV(svr, parameters)

clf.fit(train_array, label_train_final)

y_predicted = clf.predict(test_array)
#y_actual = label_test_final
print y_predicted
print label_test_final
#print clf.score(X_test, y_test)
#y_test, y_predicted = check_arrays(y_test, y_predicted)

print mean_absolute_error(label_test_final, y_predicted)
print math.sqrt(mean_squared_error(label_test_final, y_predicted))
dfypre = pd.DataFrame(y_predicted)
#dfypre.to_csv("ypre.csv", index=False)
dfyact = pd.DataFrame(label_test_final)
#dfyact.to_csv("ytes.csv", index=False)
