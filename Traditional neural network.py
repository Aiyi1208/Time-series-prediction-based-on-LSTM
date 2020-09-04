#Packages used in the import program
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import read_csv
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Nadam

#############################################################
#Import time series data#
dataframe = read_csv('windspeed1.csv',
header=0, sep=";", squeeze=True, parse_dates=True)
#############################################################

##Convert data list to matrix
dataframe = dataframe[0:1440]
dataset = dataframe.values
dataset=dataset[:,np.newaxis]
#The data were normalized
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

##Set up training set and test set
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

##Define the scroll mechanism function and set the input window value
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 8
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0],testX.shape[1]))
#################################################################

##The single hidden layer neural network with input of 8 and hidden node of 7 is configured, and Adam optimizer is adopted
model=Sequential()
model.add(Dense(output_dim=7,input_dim=8))
model.add(Dense(output_dim=1))
model.compile(loss='mse',optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
#Dropout mechanism is not used
#########################################################################################
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#################################################################################

#Structural evaluation index
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# calculate mean absolute error
trainScore = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAE' % (testScore))
import numpy as np
# calculate mean absolute percent error

def mape(y_true, y_pred):
 
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape
M1=mape(testY[0],testPredict[:,0])
M2=mape(trainY[0],trainPredict[:,0])
print('Train Score: %.4f MAE' % (M2))
print('Test Score: %.4f MAPE' % (M1))

# calculate R^2
def R(y_true, y_pred):
  
    n = len(y_true)
    a=sum(y_true)/n
    R=1-(sum((y_true-y_pred)**2)/sum((y_pred-a)**2));
    return R
R1=(trainY[0],trainPredict[:,0])
R2=R(testY[0],testPredict[:,0])
print('Train Score: %.4f R^2' % (R1))
print('Test Score: %.4f R^2' % (R2))
##########################################################
#Export forecast values to a CSV file
test=pd.DataFrame(data=testPredict[:,0])
test.to_csv('D:1.csv')
