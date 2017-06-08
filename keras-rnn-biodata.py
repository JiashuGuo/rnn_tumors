#!/usr/bin/env python 
'''

Written by: Gregory Ditzler
'''
import keras
import pickle
import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers

# -----------------------------------------------------------------------------
# setup some free parameters of the experiment 
data_dim = 2
timesteps = 32
hidden_nodes = 30
batch_size = 16
epochs = 10
verbose = 1
validation_split = .2

# -----------------------------------------------------------------------------
# 
# 
# 
# - The first ranges 1 - 5 and denotes the motility rate: number*0.001.
# - The next TWO range from 1 - 10 and denote the angiogenic rate: number*0.05.
# - The last THREE range from 1 - 100 and denote the mitotic rate: number*0.0025

mos = [1]
ans = [1]
mis = [1,15,25,50,75,100]

base_path = 'FLAIR_DATA/'
num_classes = 0
for mo in mos:
    for an in ans:
        for mi in mis:
            num_classes += 1

curr_class = 0
start = True
save_name = ''
for mo in mos:
    for an in ans:
        for mi in mis:
            # 
            file_str = base_path+str(mo)
            if an < 10:
                file_str += '0'
            file_str += str(an)
            if mi < 10:
                file_str += '0'
            if mi < 100:
                file_str += '0'
            file_str += str(mi) + '.txt'
            
            #
            if start:
                X = np.loadtxt(file_str, delimiter='\t')
                x_train = np.zeros(( np.int(X.shape[0]/timesteps), timesteps, data_dim))  
                y_train = np.zeros(( np.int(X.shape[0]/timesteps), num_classes))  
                for n in range(x_train.shape[0]):
                    x_train[n,:,:] = X[timesteps*n:timesteps*(n+1), :]
                    y_train[n, curr_class] = 1
                start = False

            else:
                # 
                X = np.loadtxt(file_str, delimiter='\t')
                x_t = np.zeros(( np.int(X.shape[0]/timesteps), timesteps, data_dim))  
                y_t = np.zeros(( np.int(X.shape[0]/timesteps), num_classes))
                for n in range(x_t.shape[0]):
                    x_t[n,:,:] = X[timesteps*n:timesteps*(n+1), :]
                    y_t[n, curr_class] = 1

                x_train = np.concatenate( (x_train, x_t), axis = 0 )
                y_train = np.concatenate( (y_train, y_t), axis = 0 )

            # 
            curr_class += 1          


# -----------------------------------------------------------------------------
'''
model = Sequential()
model.add(LSTM(hidden_nodes, 
    return_sequences=True, 
    input_shape=(timesteps, data_dim)))   
model.add(LSTM(hidden_nodes, 
    return_sequences=True))   
model.add(LSTM(hidden_nodes))   
model.add(Dense(num_classes, 
    activation='softmax'))
'''

model = Sequential()
model.add(LSTM(hidden_nodes, 
    return_sequences=True, 
    input_shape=(timesteps, data_dim)))   
model.add(LSTM(hidden_nodes, 
    dropout=0.2, 
    recurrent_dropout=0.2))
model.add(Dense(num_classes, 
    activation='softmax'))


'''
model = Sequential()
model.add(LSTM(hidden_nodes, 
    return_sequences=True, 
    input_shape=(timesteps, data_dim)))   
model.add(Dropout(0.5))
model.add(LSTM(hidden_nodes))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
'''

sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
    optimizer=sgd, 
    metrics=['accuracy'])


history = model.fit(x_train, y_train, 
    batch_size=batch_size, 
    epochs=epochs,
    shuffle=True, 
    validation_data=None, 
    validation_split=validation_split, 
    verbose=verbose)

# -----------------------------------------------------------------------------
# build a string that will save a file name that is specific to the task we 
# are performing. 
save_name = 'experiment_mo_'
for mo in mos:
    save_name += str(mo) + '-'
save_name = save_name[:-1]
save_name += '_an_'
for an in ans:
    save_name += str(an) + '-'
save_name = save_name[:-1]
save_name += '_mi_'
for mi in mis:
    save_name += str(mi) + '-'
save_name = save_name[:-1]
save_name += '_LSTM_'+str(hidden_nodes)+'_time_'+str(timesteps)+'_epoch_'+str(epochs)+'_mb_'+str(batch_size)+'.pkl'

# -----------------------------------------------------------------------------
# save the file output
params = { 'history': history, 'model': model }
pickle.dump(params, open(save_name, 'wb') )

