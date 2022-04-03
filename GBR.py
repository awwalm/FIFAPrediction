# GBR.py

print(__doc__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from numpy import asarray
from numpy import savetxt

from sklearn.utils import shuffle
from sklearn import ensemble

# skipping previously explained preprocessing steps

# read the file
df = pd.read_csv("Datasets/Outfield_Players_features.csv")
df.head

# impute the missing values
df.isnull().sum()

# missing values in the Outfield Players dataset needs to be imputed
# this little checkup (previous code block) reveals the followng missing values:
# release_clause_eur     55
# team_position          21  -> not too relevant. to be dropped later on
# dribbling              25
# passing                25
# shooting               25
# pace                   25

# imputing release_clause_eur with mean value
mean = df['release_clause_eur'].mean()
df['release_clause_eur'].fillna(mean, inplace = True)

# imputing dribbling with mean value
mean = df['dribbling'].mean()
df['dribbling'].fillna(mean, inplace = True)

# imputing passing with mean value
mean = df['passing'].mean()
df['passing'].fillna(mean, inplace = True)

# now for shooting
mean = df['shooting'].mean()
df['shooting'].fillna(mean, inplace = True)

# same is done for pace
mean = df['pace'].mean()
df['pace'].fillna(mean, inplace = True)

# note that the team_position is not really too important, 
# so it can be momentarily dropped or else we convert to numeric values
#df.drop(['team_position'], axis=1, inplace=True)

# convert categorical data into numerical data if need be
df = pd.get_dummies(df)

print(df.isnull().sum(), "\n\n")

# split data into training (80%) and test set (20%)
train, test = train_test_split(df, test_size = 0.2)
# print(test[0:1])

# save the cleaned data tocsv for future use
df.to_csv("Datasets/cleaned_dataset.csv")

# identify the data to be trained followed by labels and target (overall)
x_train = train.drop('overall', axis = 1)
y_train = train['overall']

x_test = test.drop('overall', axis = 1)
y_test = test['overall']

# import statements and preprocessing steps are skipped due to similarity #######
# begin gradient boosting by fitting GBR model_selection 	#####################

# we declare parameters by specifying the number of estimators
#	minimum samples to use, and the inbuilt loss function
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

# format everything in the params dictionary and swap their values into the model
clf = ensemble.GradientBoostingRegressor(**params)

# finally fit the model
clf.fit(x_train, y_train)

# get the score
score = clf.score(x_test, y_test)

# calculate the Mean Squared Error
mse = mean_squared_error(y_test, clf.predict(x_test))

# print our values (MSE, prediction size, score, and prediction itself)
print("MSE: %.4f" % mse)
print("size of prediction: ", len(clf.predict(x_test)))
print("prediction: \n", clf.predict(x_test))
print("test score: {0:.4f}\n".format(score))
# visualization time 	########################################################

# training deviance -> first compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()

# plot feature importance
feature_importance = clf.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, x_test.columns[sorted_idx])
plt.yticks(None)

# plt.setp(plt.subplot(1, 2, 2).get_yticklabels(), visible=False)
plt.ylabel('Important Attributes')
plt.ylim(1020, 1040) # limit the values to show since it's not much
plt.xlabel('Relative Importance')
plt.title('Variable Importance')

plt.show()
