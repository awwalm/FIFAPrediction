# MLReg.py

print(__doc__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from numpy import asarray
from numpy import savetxt


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

# we begin the multiple linear regression prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
score = regressor.score(x_test, y_test)

# Predicting the Test set results
model = LinearRegression()
pred = regressor.predict(x_test)
print("predicted values:\n\n", pred, "\n\n\n")

###################################

# compare predictions with actual values
plt.scatter(y_test, pred)
plt.xlabel('true values')
plt.ylabel('predictions')
plt.title('true values vs. predicted values @score: {0:.4f}'.format(score))
plt.show()

# no writing to csv required in this case