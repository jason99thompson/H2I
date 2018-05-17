# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:01:49 2018

@author: thompja
"""
import pandas as pd
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

# Parameters
DATA_DIR = 'C:/Users/thompja/OneDrive - Sky/201 Python/06 Programme Schedule Analysis/02 Data For Model/01 Simple Model/'
DATA_FILE = '02_simple_hhd_hr_summary_201804'
MINS_IN_HR = 5
BATCH_SIZE = 24
EPOCHS = 500


#   Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import data
barb_data = pd.read_csv(DATA_DIR + DATA_FILE, sep='|')
barb_data = barb_data.sort_values(by=['household_number',
                                             'the_month',
                                             'the_day',
                                             'the_hour'])
unique_hhds = pd.unique(barb_data.iloc[:,0])
barb_data = barb_data[(barb_data.the_day <= 14)]
dataset = barb_data.iloc[:,3:].values
y = barb_data.iloc[:,5].values
# Set target to a 0,1 value depending upon whether the 
# number of mins watched in the hr is greater than MINS_IN_HR
# y = (y >= MINS_IN_HR * 60).astype(int)


training_hhds = int(len(unique_hhds) * 0.7)
training_rows = training_hhds * 336

X_train_raw = dataset[: training_rows, :]
X_test_raw = dataset[training_rows : , :]
y_train_raw = y[: training_rows]
y_test_raw = y[training_rows : ]



# normalization the data
scaler_X = StandardScaler()
scaler_X = scaler_X.fit(dataset)
X_train = scaler_X.transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

# y is actually the last feature!!
y_train = X_train[:, 2]
y_test = X_test[:, 2]


# 14 days * 24 hrs = 336 rows per sample (hhd)
X_train = np.asarray([X_train[x * 336 : (x + 1) * 336] for x in range(training_hhds)])
X_test = np.asarray([X_test[x * 336 : (x + 1) * 336] for x in range(len(unique_hhds) - training_hhds)])
y_train = np.asarray([y_train[x * 336 : (x + 1) * 336] for x in range(training_hhds)])
y_test = np.asarray([y_test[x * 336 : (x + 1) * 336] for x in range(len(unique_hhds) - training_hhds)])



# %%

#   The Model
model = Sequential()
model.add(LSTM(32, input_shape=(336, 3)))
model.add(Dense(336))

# model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
# model.add(Dense(1, activation='sigmoid'))

#   Compile
model.compile(loss='binary_crossentropy',  
            optimizer='adam',              
            metrics=['accuracy'])

#   Train

# cbk_early_stopping = EarlyStopping(monitor='val_acc', patience=2, mode='max')
model.fit(X_train, y_train, BATCH_SIZE, epochs=EPOCHS,  
            validation_data=(X_test, y_test)) 
           # callbacks=[cbk_early_stopping] )

score, acc = model.evaluate(X_test, y_test,
                            batch_size=BATCH_SIZE)
print('test score:', score, ' test accuracy:', acc)

print(X_train.shape[1])

# %%

def fit_lstm_cat(X_train, y_train, X_test, y_test, batch_size, 
                 nb_epoch, neurons):
    """
    # fit an LSTM network to training data
    https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
    """
    # prepare model
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(X_train.shape[1]))
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    # fit model
    train_acc_history = []
    test_acc_history = []
    for i in range(nb_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=i + 1,
                  validation_data=(X_test, y_test))
        train_score, train_acc = model.evaluate(X_train, y_train,
                                                batch_size=batch_size)        
        test_score, test_acc = model.evaluate(X_test, y_test,
                                              batch_size=batch_size)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        # model.reset_states()	 
    return train_acc_history, test_acc_history


def fit_lstm_rsme(X_train, y_train, X_test, y_test, batch_size, 
                  nb_epoch, neurons):
    """
    # fit an LSTM network to training data
    https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
    """
    # prepare model
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(X_train.shape[1]))
    model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])
    # fit model
    train_acc_history = []
    test_acc_history = []
    for i in range(nb_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=i + 1,
                  validation_data=(X_test, y_test))
        train_score, train_acc = model.evaluate(X_train, y_train,
                                                batch_size=batch_size)        
        test_score, test_acc = model.evaluate(X_test, y_test,
                                              batch_size=batch_size)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        # model.reset_states()	 
    return train_acc_history, test_acc_history



# %%
    
testtrain_results_results = fit_lstm_cat(X_train, y_train, X_test, y_test, 
                            BATCH_SIZE, 2, 32)




train_results, test_results = fit_lstm_rsme(X_train, y_train, X_test, y_test, 
                             BATCH_SIZE, 50, 32)





