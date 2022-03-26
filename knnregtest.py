print(__doc__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
# %matplotlib inline

from numpy import asarray
from numpy import savetxt

# read the file
df = pd.read_csv("Outfield_Players_features.csv")
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
df.to_csv("cleaned_dataset.csv")

# identify the data to be trained followed by labels and target (overall)
x_train = train.drop('overall', axis = 1)
y_train = train['overall']

x_test = test.drop('overall', axis = 1)
y_test = test['overall']

# evaluation 2: testing vs. training accuracy with k-neighbors count

# training and test set comparison
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 19)

for n_neighbors in neighbors_settings:
    # retrain/build the model
    clf = neighbors.KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(x_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(x_train, y_train))
    # records generalization accuracy
    test_accuracy.append(clf.score(x_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
#plt.show()
plt.legend()


# we skip the scaling for now since the dataset looks fine and well scaled
# moving on to the different K-values evaluation

# array to store r mean square error values for different values of k
rmse_val = []

for k in range(20):
    k = k+1
    model = neighbors.KNeighborsRegressor(n_neighbors = k)
    model.fit(x_train, y_train) # model fitting
    pred = model.predict(x_test) # make prediction on test set
    error = sqrt(mean_squared_error(y_test, pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , k , 'is:', error)

#plotting the rmse values against k values
# the resulting double elbow curve reveals 3 and 9
# as the most consisitent K-values
curve = pd.DataFrame(rmse_val) 
curve.plot()

# sample prediction and evaluation
k = 3
model = neighbors.KNeighborsRegressor(n_neighbors = k)
model.fit(x_train, y_train) # model fitting
pred = model.predict(x_test) # prediction
evaluation = model.score(x_test, y_test) # evaluation or score
print("predictions: \n",pred, "\n")
print(y_test, "\n")
print("score: ", evaluation, "\n")

# sample visualization
from pandas.plotting import scatter_matrix
newdf = pd.DataFrame(np.random.randn(1000, 7), 
                     columns=['overall', 'value_eur', 'skill_dribbling', 'skill_ball_control', 'passing', 'shooting', 'pace'])
pd.plotting.scatter_matrix(newdf)
plt.show()

scatter_matrix(newdf, alpha=0.2, figsize=(7,7), diagonal='kde')
#######################################################################
# time for real deal prediction
test.to_csv('test.csv')
y_test.to_csv('y_test.csv')

predictions = pd.DataFrame(pred, columns = ['overall_pred'])
predictions.to_csv('pred.csv', index = False)

# with basic printing done, we do this properly and output it to a file
new_test = pd.read_csv('test.csv')
submission = pd.read_csv('SampleSubmission.csv')
submission['sofifa_id'] = new_test['sofifa_id']
submission['age'] = new_test['age']
submission['actual_overall'] = new_test['overall']

# preprocess the test csv file
new_test.drop(['sofifa_id', 'age'], axis = 1, inplace = True)
# new_test['overall'].fillna(mean, inplace = True)
# new_test = pd.get_dummies(new_test)

# predicting on the test set and creating submission file
predict = model.predict(new_test)
submission['overall'] = predict
submission.to_csv('submit_file.csv', index = False)

# evaluation 1: multiple KNN Regression predictions with various k values

fig, axes = plt.subplots(1, 2, figsize = (15, 4))
plt.suptitle("nearest neighbor regression")

# plot the comparison graph for the specified KNeighbors
line = np.linspace(-3, 3, 200).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 5, 7, 9, 11, 13, 15, 17], axes):
    reg = neighbors.KNeighborsRegressor(n_neighbors = n_neighbors).fit(x_test, y_test)
    ax.plot(x_test,y_test, 'o')
    ax.plot(x_test, -3 * np.ones(len(x_test)), 'o')
    ax.plot(line, reg.predict(x_test))
    ax.set_title("%d neighbor(s)" % n_neighbors)


