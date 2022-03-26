# VotingReg.py

print(__doc__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import cross_val_predict
from numpy import asarray
from numpy import savetxt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor


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

# we begin the voting regression algorithm using RandomForest and LinearRegression
reg1 = RandomForestRegressor(random_state = 0, n_estimators = 10)
reg2 = LinearRegression()
vreg = VotingRegressor([ ('rf', reg1), ('lr', reg2) ])

# fit the additive models
reg1.fit(x_train, y_train)
reg2.fit(x_train, y_train)
vreg.fit(x_train, y_train)

# score and check the accuracy
rfscore = reg1.score(x_test, y_test)
lrscore = reg2.score(x_test, y_test)
vrscore = vreg.score(x_test, y_test)

# make predictions and print intermediate results
print("RandomForestRegressor Predictions:\n", reg1.predict(x_test), "\n")
print("Linear Regression Predictions:\n", reg2.predict(x_test), "\n")
print("Voting Regressor Predictions:\n", vreg.predict(x_test), "\n")

# visualization and comparison
plt.figure()
plt.plot(reg1.predict(x_test), 'b^', label = 'RandomForestRegressor @score {0:.2f}'.format(rfscore))
plt.plot(reg2.predict(x_test), 'ys', label = 'LinearRegression @score {0:.2f}'.format(lrscore))
plt.plot(vreg.predict(x_test), 'r*', label = 'VotingRegressor @score {0:.2f}'.format(vrscore))
# plt.tick_params(axis = 'x', which = 'both', top = False, labelbottom = True)
plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions with averaged')
plt.show()

# recommended writing to csv since there're multiple additive predictors
actual_col = np.array(y_test)
rf_col = np.array(reg1.predict(x_test))
lr_col = np.array(reg2.predict(x_test))
vr_col = np.array(vreg.predict(x_test))
save_df = pd.DataFrame(
{'actual_overall':actual_col, 'rforest_reg_predicted':rf_col, 'linear_reg_predicted':lr_col, 'voting_reg_predicted':vr_col})
save_df.to_csv("Datasets/VotingRegressorSubmission.csv", index = False)