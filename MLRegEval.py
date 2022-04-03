# MLRegEval.py

print(__doc__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from numpy import asarray
from numpy import savetxt
from sklearn.linear_model import Ridge


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

print(df.isnull().sum())

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

#-------------------------------------------------------

# formal evaluation 1: RidgeRegression

# default with alpha/shrinkage at 1.0
ridge = Ridge().fit(x_train, y_train)
ridge_def_score = ridge.score(x_test, y_test)

# with alpha/shrinkage parameter set to 10.0
ridge10 = Ridge(alpha = 10).fit(x_train, y_train)
ridge10_score = ridge10.score(x_test, y_test)

# with alpha at 0.1
ridge01 = Ridge(alpha = 0.1).fit(x_train, y_train)
ridge01_score = ridge01.score(x_test, y_test)

# plain Multiple Linear Regression
lr = LinearRegression().fit(x_train,y_train)
lr_score = lr.score(x_test, y_test)

# visualize the differences
plt.title("ridge_coefficients")
plt.plot(ridge.coef_, 'o', label="Ridge alpha=1 @score: {0:.4f}".format(ridge_def_score)) 
plt.plot(ridge10.coef_, 'o', label="Ridge alpha=10 @score: {0:.4f}".format(ridge10_score)) 
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1 @score: {0:.4f}".format(ridge01_score))
plt.plot(lr.coef_, 'o', label="Multiple Linear Regression @score: {0:.4f}".format(lr_score)) 
#plt.ylim(-25, 25)
plt.legend()

# -------------------------------------------------------

# we begin the multiple linear regression prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
model = LinearRegression()
pred = regressor.predict(x_test)
print("predicted values:\n\n", pred, "\n\n\n")

#######################################################################################
# cross -validation evaluation

# print the score
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score: ", score, "\n")

# applying cross validation measurement at an interval of 10
pred2 = cross_val_predict(model, x_test, y_test, cv=10)
print("cross validation measured values\n", pred2, "\n")

# now show the validated values at every 10th interval
cross_val_scores = cross_val_score(model, x_test, y_test, cv=10)
print ("cross validation scores at every 10th interval:\n", cross_val_scores, "\n")

# rounding it up with visualzation
fig, ax = plt.subplots()
ax.scatter(y_test, pred2, edgecolors=(0,0,0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('cross validation')
plt.show()

# compare predictions with actual values
plt.scatter(y_test, pred)
plt.xlabel('true values')
plt.ylabel('predictions')
plt.title('true values vs. predicted values @score: {0:.4f}'.format(score))
plt.show()

# no writing to csv required in this case