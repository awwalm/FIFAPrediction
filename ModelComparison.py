# ModelComparison.py

print(__doc__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

# the models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor #needed for Voting Regressor

# read the file
df = pd.read_csv("Datasets/Outfield_Players_features.csv")
df.head

# impute the missing values
df.isnull().sum()

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

# convert categorical data into numerical data if need be
df = pd.get_dummies(df)

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

# prepare voting regressor weak builders
reg1 = RandomForestRegressor(random_state = 0, n_estimators = 10)
reg2 = LinearRegression()

# prepare gradient boosting regressor parameters
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

# identifying random seed
random_seed = 12

# preparing the models to use
outcome = []          # results of cross validation
model_names = []      # the name tag of the models
models = []           # tuple mapping for DECLARING the models

# insert the models inside the respective array
models.append(('KNNReg', KNeighborsRegressor(n_neighbors = 9)))
models.append(('MLR', LinearRegression()))
models.append(('VReg', VotingRegressor([ ('rf', reg1), ('lr', reg2) ])))
models.append(('GBR', GradientBoostingRegressor(**params)))

# run the K-fold analysis through a for loop on all models
# and generate mean and standard deviation for all models
for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits = 10, random_state = random_seed, shuffle = True)
    results = model_selection.cross_val_score(model, x_test, y_test, cv = k_fold_validation, scoring = 'r2')
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean = %f STD = %f Variance = %f" % (model_name, results.mean(), results.std(), results.var())
    print(output_message)
    
# finally, visualize the analysis
fig = plt.figure()
fig.suptitle('FIFA Ratings Prediction Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
ax.set_xticklabels(model_names)
plt.show()
