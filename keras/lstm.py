# lstm model
import time
import random
import numpy as np
from numpy import array
from numpy import mean
from numpy import std
from numpy import argmax
from pandas import read_csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.metrics import Precision, Recall, AUC, MeanAbsoluteError
from sklearn.metrics import f1_score, matthews_corrcoef
from keras.utils import to_categorical

# Seed
seed = 166
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_data(file):
    loaded = list()
    chunk_size = 16  
    chunk_iterator = read_csv(file, chunksize=chunk_size)

    for chunk in chunk_iterator:
        chunk = chunk.iloc[:, :20]  
        loaded.append(chunk)

    return loaded

# def load_data(file):
#     chunk_size = 16  
#     total_data = read_csv(file)
#     loaded = []

#     for i in range(len(total_data) - chunk_size + 1):
#         chunk = total_data.iloc[i:i+chunk_size, :20]
#         loaded.append(chunk)

#     return loaded

# load data
def data_load(n):
    filepath = "../Data/time_series/k-fold/whole_block/"
    train_X = f'train_x_{n}.csv'
    train_Y = f'train_y_{n}.csv'
    train_file = filepath + train_X

    # load train data
    trainX = load_data(train_file)
    # trainX = transpose(trainX, (2, 0, 1))
    trainX = array(trainX)
    trainy = read_csv(filepath + train_Y, delim_whitespace=True)
    trainy = to_categorical(trainy)
    print('train shape:',trainX.shape, trainy.shape)

    # load test data
    test_X = f'test_x_{n}.csv'
    test_Y = f'test_y_{n}.csv'
    test_file = filepath + test_X

    testX = load_data(test_file)
    # testX = transpose(testX, (2, 0, 1))
    testX = array(testX)
    testy = read_csv(filepath + test_Y, delim_whitespace=True)
    # print('test shape:',testX.shape, testy.shape)
    testy = to_categorical(testy)

    print('\nDataset loaded no.', n)
    print('shapes:', trainX.shape, trainy.shape, testX.shape, testy.shape)
    # print('exp_', n)

    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, n):
 verbose, epochs, batch_size = 2, 200, 64
 n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
 model = Sequential()
 model.add(LSTM(64, input_shape=(n_timesteps,n_features), return_sequences=True))
 model.add(Dropout(0.5))

 model.add(LSTM(32, return_sequences=True))
 model.add(Dropout(0.5))

#  model.add(LSTM(128, return_sequences=True))
#  model.add(Dropout(0.5))

#  model.add(LSTM(256, return_sequences=True))
#  model.add(Dropout(0.5))

 model.add(LSTM(16))
 model.add(Dropout(0.5))

 model.add(Dense(16, activation='relu'))
 model.add(Dense(n_outputs, activation='softmax'))

 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall(), AUC(), MeanAbsoluteError()])

 # fit network
 history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

 # evaluate model
 loss, accuracy, precision, recall, auc, mae = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

#  print(f"Train loss: {history.history['loss'][-1]:.8f}, Test loss: {loss:.8f}")
 print('Train info:')
 losses = history.history['loss']
 accuracies = history.history['accuracy']
 for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs} - Loss: {losses[epoch]:.4f}, Accuracy: {accuracies[epoch]:.4f}')

 # Make predictions on test set
 y_pred = model.predict(testX)
 y_true = argmax(testy, axis=1)
 y_pred = argmax(y_pred, axis=1)

# Calculate F1-Score and MCC
 f1 = f1_score(y_true, y_pred, average='weighted')
 mcc = matthews_corrcoef(y_true, y_pred)

# Save the entire model to a file.
 model.save(f'Model_lstm3_{n}w.keras')

# Save only the model weights to a file.
 model.save_weights(f'Model_lstm3_{n}.keras')

#  return accuracy
 return accuracy, precision, recall, auc, f1, mcc, mae  
 
# summarize scores
def summarize_results(scores):
 metrics = ["accuracy", "precision", "recall", "auc", "f1", "mcc", "mae"]
 scores = array(scores)
 result = {}

 for idx, metric in enumerate(metrics):
    m = mean(scores[:, idx])
    s = std(scores[:, idx])
    result[metric] = (m, s)  
 return result

# repeat experiment
for n in range(0,10):
    trainX, trainy, testX, testy = data_load(n)
    repeats = 10
    scores = list()
    accuracy = list()
    start_time = time.time()

    for r in range(repeats):

        score = evaluate_model(trainX, trainy, testX, testy, n)   
        # print('>#%d: %.3f' % (r+1, score))

        print('>#', r+1 ,': acc %.4f%%' % (score[0]*100.0), 'prec %.4f' % (score[1]), 'recall %.4f' % (score[2]), \
            'f1 %.4f' % (score[4]), 'mcc %.4f' % (score[6]) , 'mae %.4f' % (score[6] / 100.0))
        accuracy.append(score[0])
        scores.append(score)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The operation took {elapsed_time} seconds.")

    # summarize results
    results = summarize_results(scores)
    for metric, (m, s) in results.items():
        print(f"{metric}: Mean = {m:.4f}, Std = {s:.6f}")
