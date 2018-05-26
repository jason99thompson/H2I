# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:01:49 2018

@author: thompja
"""
# ############################## Imports ######################################

# %matplotlib

import time
import datetime
import ggplot
import matplotlib.pyplot as plt

import pandas as pd
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV

import numpy as np
import scipy 

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.callbacks import TensorBoard
import keras.backend as K

import tensorflow as tf
# ############################## Parameters ###################################
DATA_DIR = 'C:/Users/thompja/OneDrive - Sky/201 Python/06 Programme Schedule Analysis/02 Data For Model/01 Simple Model/'
DATA_FILE = '02_simple_hhd_hr_summary_201804'
LOG_DIR = 'C:/Users/thompja/OneDrive/103 Barb Modelling/Models/Logs/'
# cd C:/Users/thompja/OneDrive/103 Barb Modelling/Models
# tensorboard --logdir=logs/
MINS_IN_HR = 5
BATCH_SIZE = 480
EPOCHS = 2
# train on first number of days of April 2018
TRAINING_DAYS = 28

# Metrics to track
continuous = ['mse', 'mape']
# cat8 = ['acc', zero_acc, nonzero_acc]

# ############################## Environment setup ############################

#   Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ############################## Import Data ##################################

# import data and sort
barb_data_raw = pd.read_csv(DATA_DIR + DATA_FILE, sep='|')
barb_data_raw = barb_data_raw.sort_values(by=['household_number',
                                             'the_month',
                                             'the_day',
                                             'the_hour'])

# ############################## Process Data #################################

# array of unique household_number
unique_hhds = pd.unique(barb_data_raw.iloc[:,0])
# Just using 1st TRAINING_DAYS of April
barb_data = barb_data_raw[(barb_data_raw.the_day <= TRAINING_DAYS)]
# last column of dataset is number of mins watched
dataset = barb_data.iloc[:,3:].values
y = barb_data.iloc[:,5].values 
# One-hot code into 8 duration categories zero and 10min intervals
# y = y_8cats(y)
# Binary category, zero or > zero mins
y = (y > 0).astype(int)


# Take 70% of data to train on and 30% to test
# Each household will have 24 rows per day
training_hhds = int(len(unique_hhds) * 0.7)
training_rows = training_hhds * TRAINING_DAYS * 24
# Split data into training & test
X_train_raw = dataset[: training_rows, :]
X_test_raw = dataset[training_rows : , :]
# y_train_raw = y[: training_rows]
# y_test_raw = y[training_rows : ]

# normalization the data. Want all features to be bewteen 0 and 1
scaler_X = MinMaxScaler()
scaler_X = scaler_X.fit(dataset)
X_train = scaler_X.transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)
# y is actually the last feature!!
# so exclude from X
#y_test = X_test[:, 2]
#y_train = X_train[:, 2]
y_train = y[: training_rows]
y_test = y[training_rows : ]
y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
X_train = X_train[:, : 2]
X_test = X_test[:, : 2]



# %%
# ############################## Set up & Run Model ###########################

# ---------------------------------------------------
# Activation Functions: ftp://ftp.sas.com/pub/neural/FAQ2.html#A_act
# For continuous-valued targets with a bounded range, the logistic and tanh functions can be used
# 

K.clear_session()

# The Model based upon Dense layers
model = Sequential()
# inputs have have 2 features and will output to 32 neurons
model.add(Dense(8, input_shape=(2, ), activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='softmax'))
#model.add(Dense(8, activation='softmax'))
#model.add(Dense(8, activation='softmax'))
#model.add(Dense(8, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

#   Compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[F1_score])

# tensorboard = TensorBoard(log_dir=LOG_DIR + '/01_Simple_' + now_string())
#, histogram_freq=1,
#                         batch_size=BATCH_SIZE, write_graph=True, write_grads=True, 
#                         write_images=True,
#                         embeddings_freq = 1)
# , embeddings_layer_names = [], embeddings_metadata = [])

#   Train
# cbk_early_stopping = EarlyStopping(monitor='val_acc', patience=2, mode='max')
history = model.fit(X_train, y_train, BATCH_SIZE, epochs=EPOCHS,  
                    validation_data=(X_test, y_test))
           # callbacks=[tensorboard]) 
           # callbacks=[cbk_early_stopping] )

# %%
# ############################## Loop Through Model ###########################

history = batch_loop(X_train, y_train, X_test, y_test, 
                     batch_size_start=10, batch_size_end=1000, 
                     batch_step=10, epochs=2)

history_train = history[0]
history_test = history[1]


train_hist, test_hist = learning_curve(barb_data_raw, BATCH_SIZE, EPOCHS, 28)






# %%
# ############################## QA Custom Metrics ############################

test_predict = model.predict(X_test, batch_size=BATCH_SIZE, 
                             verbose=1, steps=None)
# test_predict_classes = model.predict_classes(X_test)
# test_predict_probs = model.predict_proba(X_test)


# calculate F1
# precision
test_predict_01 = (test_predict >= 0.5).astype(int)
y_test_reshape = np.reshape(y_test, (y_test.shape[0], 1))
true_pos = y_test_reshape * test_predict_01
true_pos = np.sum(y_test_reshape * test_predict_01)
pred_pos = np.sum(test_predict_01)
P = true_pos / pred_pos
# recall
true_act = np.sum(y_test_reshape)
R = true_pos / true_act
F1 = 2 * (P * R / (P + R + 1e-12))
print('true_pos: ',true_pos, ' pred_pos: ', pred_pos, ' true_act: ', true_act )

#%%
# ############################## Plots ########################################

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['F1_score'])
plt.plot(history.history['val_F1_score'])
plt.title('Model F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='middle right')
plt.show()

print(history.history['F1_score'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

F1_history_train = history.history['F1_score']
F1_history_test = history.history['val_F1_score']
loss_train = history.history['loss']
loss_test = history.history['val_loss']


#%%
# ############################## Plots ########################################

             
# %%

score_train, acc_train = model.evaluate(X_train, y_train,
                                        batch_size=BATCH_SIZE)

score_test, acc_test = model.evaluate(X_test, y_test,
                                      batch_size=BATCH_SIZE)

print('train score:', score_train, ' train accuracy:', acc_train)
print('test score:', score_test, ' test accuracy:', acc_test)

# train_predict = model.predict(X_train, batch_size=BATCH_SIZE, verbose=1, steps=None)
test_predict = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1, steps=None)
# calculate F1




# labels = np.zeros(y_train.shape)
# scatter_with_colours(y_train, train_predict, labels)
# print(scipy.stats.pearsonr(y_test.reshape((576912)), test_predict.reshape((576912))))


# model.save('C:/Users/thompja/OneDrive/103 Barb Modelling/Models/H2I/NN_01')


model.summary()

# from keras.utils import plot_model
# import pydot
# import graphviz
# plot_model(model, to_file='LOG_DIR/' + '/01_Simple_' + now_string() + '.png')





# %%
# ############################## Metrics ######################################


def true_pos(y_true, y_pred):
     # assign 0-1 to y_pred. Values >= 0.5 get assigned non-zero i.e. 1 
    cut_off = K.zeros_like(y_pred) + 0.5
    y_pred_01 = K.cast(K.greater_equal(y_pred, cut_off), 'float32')
    # number of true positives
    true_pos = K.sum(y_true * y_pred_01)
    return true_pos


def pred_pos(y_true, y_pred):
    # assign 0-1 to y_pred. Values >= 0.5 get assigned non-zero i.e. 1 
    cut_off = K.zeros_like(y_pred) + 0.5
    y_pred_01 = K.cast(K.greater_equal(y_pred, cut_off), 'float32')
    # number of predicted positives 
    pred_pos = K.sum(y_pred_01)
    return pred_pos

def true_act(y_true, y_pred):
    return K.sum(y_true)



def precision(y_true, y_pred):
    """
    Of all samples where we predicted y = 1, what fraction actually is a 1
    true positives / no. of predicted positive
    
    """
    # assign 0-1 to y_pred. Values >= 0.5 get assigned non-zero i.e. 1 
    cut_off = K.zeros_like(y_pred) + 0.5
    y_pred_01 = K.cast(K.greater_equal(y_pred, cut_off), 'float32')
    # number of true positives
    true_pos = K.sum(y_true * y_pred_01)
    # number of predicted positives 
    pred_pos = K.sum(y_pred_01)
    return true_pos / (pred_pos + 1e-12)
    

def recall(y_true, y_pred):
    """
    Of all samples that are 1, what fraction did we correctly predict as 1
    true positives / no. of actual positive
    """
    # assign 0-1 to y_pred. Values >= 0.5 get assigned non-zero i.e. 1 
    cut_off = K.zeros_like(y_pred) + 0.5
    y_pred_01 = K.cast(K.greater_equal(y_pred, cut_off), 'float32')
    # number of true positives
    true_pos = K.sum(y_true * y_pred_01)
    # number of actual positives
    true_act = K.sum(y_true)
    return true_pos / (true_act + 1e-12)


def F1_score(y_true, y_pred):
    """
    F1 = 2 * (P * R / (P + R))
    """
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    return 2 * (P * R / (P + R + 1e-12))


def bin_zero_acc(y_true, y_pred):
    """
    For binary classification
    % of zero mins category (== 0) predicted correctly
    """
    # Zero = 0 and Non-Zero = 1. Switch these around for y_true
    true_all_zeros = K.zeros_like(y_true)
    y_true_switch = K.cast(K.equal(y_true, true_all_zeros),'float32')
    true_sum = K.sum(y_true_switch)
    # now assign 0-1 to y_pred. Values < 0.5 get assigned zero i.e. 1
    cut_off = K.zeros_like(y_pred) + 0.5
    y_pred_01 = K.cast(K.less(y_pred, cut_off), 'float32')   
    # number of zero true that also predicted zero
    pred_sum = K.sum(y_pred_01 * y_true_switch)
    # nan returned for validation without + 1e-12
    return pred_sum / (true_sum + 1e-12)


def bin_nonzero_acc(y_true, y_pred):
    """
    For binary classification
    % of non-zero mins category (== 1) predicted correctly
    """
    true_sum = K.sum(y_true)
    # turn y_pred into 0-1 values
    cut_off = K.zeros_like(y_pred) + 0.5    
    y_pred_01 = K.cast(K.greater_equal(y_pred, cut_off), 'float32')
    # number of non-zero true that also predicted non-zero                 
    pred_sum = K.sum(y_pred_01 * y_true)
    # nan returned for validation without + 1e-12
    return pred_sum / (true_sum + 1e-12)



def zero_acc(y_true, y_pred):
    """
    % of zero mins category (feature 0 == 1) predicted correctly
    """
    true_sum = K.sum(tf.slice(y_true, [0, 0], [-1, 1]))
                            
    # pred_sum = tf.slice(y_pred, [:, 0], [4, y_pred.get_shape()[1]-2])
    # tf.slice(foo, [3, 0], [4, foo.get_shape()[1]-2])
    #pred_sum = np.sum(y_true[:, 0] * y_pred[:,0])
    #return pred_sum / true_sum
    return true_sum



def nonzero_acc(y_true, y_pred):
    """
    % of nonzero mins category predicted correctly
    """
    true_sum = np.sum(y_true[:, 1:], axis=1)
    pred_sum = np.sum(y_pred[:, 1:], axis=1)   
    pred_sum = np.sum(true_sum * pred_sum)
    return pred_sum / true_sum




# %%
# ############################## Useful Functions #############################

def now_string():
    """
    Returns a string that represents the time now in format
    YYYYMMDD_HHMMSS e.g. '20180518_182247'        
    """
    timestamp_string = (datetime.datetime.now().strftime("%Y") + 
                       datetime.datetime.now().strftime("%m") + 
                       datetime.datetime.now().strftime("%d") + '_' +
                       datetime.datetime.now().strftime("%H") +
                       datetime.datetime.now().strftime("%M") +
                       datetime.datetime.now().strftime("%S"))
    return timestamp_string



def y_8cats(y_raw):
    """
    Create 8 duration based categories 0 mins, <=10 mins ... <=60 mins, >60mins
    """
    y = np.reshape((y_raw == 0).astype(int), (y_raw.shape[0],1))
    for i in range(10, 61, 10):      
        y_add = np.reshape(np.logical_and((y_raw > (i-10) * 60),
                                          (y_raw <= i * 60)
                                          ).astype(int),(y_raw.shape[0],1))
        y = np.concatenate((y, y_add), axis=1)
    # Multi TV households can have more than 60mins in an hour
    y_add = np.reshape((y_raw > 60 * 60).astype(int), (y_raw.shape[0],1))
    y = np.concatenate((y, y_add), axis=1)    
    return y    
    
def scatter_with_colours(x, y, labels):
    """
    Description:
        Plots a scatter plot of the vector y aginst the vector x, and colours 
        each point based upon the labels value
    
    Imports:
        import numpy as np
        import matplotlib.pyplot as plt
    
    x:
        vector plotted on x-axis
   
    y:
        vector plotted on y-axis, same dimensions as x 
   
    labels:
        the label of each data point, same dimensions as x
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Get unique list of labels
    unique_labels = np.unique(labels)
    
    
    for lab in unique_labels:
        ax.scatter(x[labels==lab], y[labels==lab],label=lab)  
        
    # plt.xlabel('X Label')
    # plt.ylabel('Y Label')
    
    ax.legend(fontsize='small')
    


# %%
# ############################## Model Loop Defs #############################

def learning_curve(data, batch_size, epochs, nb_days):
    """
    """
    K.clear_session()    
    # The Model based upon Dense layers
    model = Sequential()
    # inputs have have 2 features
    model.add(Dense(8, input_shape=(2, ), activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    #   Compile
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[F1_score])

    train_acc_history = []
    test_acc_history = []
    for dd in range(1, nb_days + 1):
        print('Day: ', dd)
        # array of unique household_number
        unique_hhds = pd.unique(data.iloc[:,0])
        # Just using 1st dd of April
        reduced_data = data[(data.the_day <= dd)]
        # last column of dataset is number of mins watched
        dataset = reduced_data.iloc[:,3:].values
        y = reduced_data.iloc[:,5].values 
        # Binary category, zero or > zero mins
        y = (y > 0).astype(int)        
        
        # Take 70% of data to train on and 30% to test
        # Each household will have 24 rows per day
        training_hhds = int(len(unique_hhds) * 0.7)
        training_rows = training_hhds * dd * 24
        # Split data into training & test
        X_train_raw = dataset[: training_rows, :]
        X_test_raw = dataset[training_rows : , :]
        
        # normalization the data. Want all features to be bewteen 0 and 1
        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(dataset)
        X_train = scaler_X.transform(X_train_raw)
        X_test = scaler_X.transform(X_test_raw)
        # y is actually the last feature!! so exclude from X
        y_train = y[: training_rows]
        y_test = y[training_rows : ]
        X_train = X_train[:, : 2]
        X_test = X_test[:, : 2]
        
        # run model 
        model.fit(X_train, y_train, batch_size, epochs=epochs,  
                  validation_data=(X_test, y_test))
        train_score, train_F1 = model.evaluate(X_train, y_train,
                                               batch_size=batch_size)        
        test_score, test_F1 = model.evaluate(X_test, y_test,
                                             batch_size=batch_size)
        train_acc_history.append(train_F1)
        test_acc_history.append(test_F1)
        model.reset_states()
    return train_acc_history, test_acc_history 


def batch_loop(X_train, y_train, X_test, y_test, 
               batch_size_start, batch_size_end, batch_step, epochs):
    """
    

    """
                
    #  >>>>>>>>>>> The Model goes here
    K.clear_session()    
    # The Model based upon Dense layers
    model = Sequential()
    # inputs have have 2 features
    model.add(Dense(8, input_shape=(2, ), activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    #   Compile
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[F1_score])
    
    # fit model
    train_acc_history = []
    test_acc_history = []
    for i in range(batch_size_start, batch_size_end + 1, batch_step):
        history = model.fit(X_train, y_train, batch_size=i, epochs=epochs,
                            validation_data=(X_test, y_test))
        train_score, train_acc = model.evaluate(X_train, y_train,
                                                batch_size=i)        
        test_score, test_acc = model.evaluate(X_test, y_test,
                                              batch_size=i)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        model.reset_states()
        print('Batch: ' + str(i), ' Train Acc: ', 
              train_acc, 'Test Acc: ', test_acc)
        model.reset_states()
    return train_acc_history, test_acc_history





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
    return test_acc


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
    return test_acc








    




