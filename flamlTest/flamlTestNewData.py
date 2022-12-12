import numpy as np
import pandas as pd
from collections import Counter
import glob
import statistics
import sklearn
import flaml


# import data

# test data
filePath = 'C:/flamlMultiOutput/BlogFeedback/test'
allFiles = glob.glob(filePath + "/*.csv")
dataFrames = pd.concat([pd.read_csv(f, header=None) for f in allFiles], ignore_index = True)


# import training data
data = pd.read_csv('C:/flamlMultiOutput/BlogFeedback/train/blogData_train.csv', header=None)

# create input and output
y_train = data.iloc[:, lambda data: [50, 51, 53, 55, 56, 58]]
X_train = data.drop(data.columns[[50, 51, 53, 55, 56, 58]], axis = 1)
y_test = dataFrames.iloc[:, lambda dataFrames: [50, 51, 53, 55, 56, 58]]
X_test = dataFrames.drop(data.columns[[50, 51, 53, 55, 56, 58]], axis = 1)



from flaml import AutoML
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# first model

setting1 = {"metric": 'mae', "time_budget": 600}
setting2 = {"metric": 'mae', "time_budget": 120}

y1 = y_train.iloc[:, [0]].squeeze()
automl1 = AutoML()
automl1.fit(X_train, y1, task="regression", **setting1)


# second model
y2 = y_train.iloc[:, [1]].squeeze()
automl2 = AutoML()
automl2.fit(X_train, y2, task="regression", **setting2)


# third model
y3 = y_train.iloc[:, [2]].squeeze()
automl3 = AutoML()
automl3.fit(X_train, y3, task="regression", **setting2)

# fourth model
y4 = y_train.iloc[:, [3]].squeeze()
automl4 = AutoML()
automl4.fit(X_train, y4, task="regression", **setting2)

# fifth model
y5 = y_train.iloc[:, [4]].squeeze()
automl5 = AutoML()
automl5.fit(X_train, y5, task="regression", **setting2)

# sixth model
y6 = y_train.iloc[:, [5]].squeeze()
automl6 = AutoML()
automl6.fit(X_train, y6, task="regression", **setting2)


y_pred1 = automl1.predict(X_test)
y_pred2 = automl2.predict(X_test)
y_pred3 = automl3.predict(X_test)
y_pred4 = automl4.predict(X_test)
y_pred5 = automl5.predict(X_test)
y_pred6 = automl6.predict(X_test)

y_test1 = y_test.iloc[:, [0]].squeeze()
y_test2 = y_test.iloc[:, [1]].squeeze()
y_test3 = y_test.iloc[:, [2]].squeeze()
y_test4 = y_test.iloc[:, [3]].squeeze() 
y_test5 = y_test.iloc[:, [4]].squeeze()
y_test6 = y_test.iloc[:, [5]].squeeze()


mae_score1 = mean_absolute_error(y_test1, y_pred1)
mae_score2 = mean_absolute_error(y_test2, y_pred2)
mae_score3 = mean_absolute_error(y_test3, y_pred3)
mae_score4 = mean_absolute_error(y_test4, y_pred4)
mae_score5 = mean_absolute_error(y_test5, y_pred5)
mae_score6 = mean_absolute_error(y_test6, y_pred6)

mse_score1 = mean_squared_error(y_test1, y_pred1)
mse_score2 = mean_squared_error(y_test2, y_pred2)
mse_score3 = mean_squared_error(y_test3, y_pred3)
mse_score4 = mean_squared_error(y_test4, y_pred4)
mse_score5 = mean_squared_error(y_test5, y_pred5)
mse_score6 = mean_squared_error(y_test6, y_pred6)

r2_score1 = r2_score(y_test1, y_pred1)
r2_score2 = r2_score(y_test2, y_pred2)
r2_score3 = r2_score(y_test3, y_pred3)
r2_score4 = r2_score(y_test4, y_pred4)
r2_score5 = r2_score(y_test5, y_pred5)
r2_score6 = r2_score(y_test6, y_pred6)


print(f'Evaluation on test data Total Comments before Basetime : MAE: {mae_score1}, MSE: {mse_score1}, R2: {r2_score1}')
print(f'Evaluation on test data Total Comments in last 24 hours before Basetime : MAE: {mae_score2}, MSE: {mse_score2}, R2: {r2_score2}')
print(f'Evaluation on test data Total Comments in first 24 hours before Basetime : MAE: {mae_score3}, MSE: {mse_score3}, R2: {r2_score3}')
print(f'Evaluation on test data Total Links before Basetime : MAE: {mae_score4}, MSE: {mse_score4}, R2: {r2_score4}')
print(f'Evaluation on test data Total Links in last 24 hours before Basetime : MAE: {mae_score5}, MSE: {mse_score5}, R2: {r2_score5}')
print(f'Evaluation on test data Total Links in first 24 hours before Basetime : MAE: {mae_score6}, MSE: {mse_score6}, R2: {r2_score6}')


# import matplotlib.pyplot as plt


# # save the models
# import pickle
# with open('automl1.pkl', 'wb') as f:
#     pickle.dump(automl1, f, pikle.HIGHEST_PROTOCOL)

# with open('automl2.pkl', 'wb') as f:
#     pickle.dump(automl2, f, pikle.HIGHEST_PROTOCOL)

# with open('automl3.pkl', 'wb') as f:
#     pickle.dump(automl3, f, pikle.HIGHEST_PROTOCOL)

# with open('automl4.pkl', 'wb') as f:
#     pickle.dump(automl4, f, pikle.HIGHEST_PROTOCOL)