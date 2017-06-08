
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np



data_dim = 2
timesteps = 16
hidden_nodes = 16
batch_size=32
epochs=10
verbose=1
validation_split = .5

'''
(1) The first ranges 1 - 5 and denotes the motility rate: number * 0.001.
(2) The next TWO range from 1 - 10 and denote the angiogenic rate: number * 0.05.
(3) The last THREE range from 1 - 100 and denote the mitotic rate: number * 0.0025
'''
mos = [x+1 for x in range(5)]
ans = [1,5,10]
mis = [1,25,50,75,100]

base_path = 'FLAIR_DATA/'
num_classes = 0
for mo in mos:
    for an in ans:
        for mi in mis:
            num_classes += 1

#plt.figure()

curr_class = 0
start = True
for mo in mos:
    for an in ans:
        for mi in mis:
            file_str = base_path+str(mo)
            if an < 10:
                file_str += '0'
            file_str += str(an)
            if mi < 10:
                file_str += '0'
            if mi < 100:
                file_str += '0'
            file_str += str(mi) + '.txt'
            
            if start:
                X = np.loadtxt(file_str, delimiter='\t')
                x_train = np.zeros(( np.int(X.shape[0]/timesteps), timesteps, data_dim))  
                y_train = np.zeros(( np.int(X.shape[0]/timesteps), num_classes))  
                for n in range(x_train.shape[0]):
                    x_train[n,:,:] = X[timesteps*n:timesteps*(n+1), :]
                    y_train[n, curr_class] = 1
                start = False

            else:
                X = np.loadtxt(file_str, delimiter='\t')
                x_t = np.zeros(( np.int(X.shape[0]/timesteps), timesteps, data_dim))  
                y_t = np.zeros(( np.int(X.shape[0]/timesteps), num_classes))
                for n in range(x_t.shape[0]):
                    x_t[n,:,:] = X[timesteps*n:timesteps*(n+1), :]
                    y_t[n, curr_class] = 1

                x_train = np.concatenate( (x_train, x_t), axis = 0 )
                y_train = np.concatenate( (y_train, y_t), axis = 0 )
    
            curr_class += 1          


def perplexity(y_true, y_pred, mask=None):
    if mask is not None:
        y_pred /= keras.sum(y_pred, axis=-1, keepdims=True)
        mask = keras.permute_dimensions(keras.reshape(mask, y_true.shape[:-1]), (0, 1, 'x'))
        truth_mask = keras.flatten(y_true*mask).nonzero()[0]  ### How do you do this on tensorflow?
        predictions = keras.gather(y_pred.flatten(), truth_mask)
        return keras.pow(2, keras.mean(-keras.log2(predictions)))
    else:
        return keras.pow(2, keras.mean(-keras.log2(y_pred)))


model = Sequential()
model.add(LSTM(hidden_nodes, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(hidden_nodes, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(hidden_nodes))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', perplexity])


model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True, 
          validation_data=None, 
          validation_split=validation_split, 
          verbose=1)


