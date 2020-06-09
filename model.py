# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:58:36 2019

@author: Adithya Kommini
"""
import os
import pickle
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from preprocess import dir_to_spectrogramMLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from modelSave import modelSave

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def cnnModel(shapeInput, noClasses):
    ## Convolution filters
    createCnn = Sequential()
    createCnn.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=shapeInput))
    createCnn.add(BatchNormalization())
    createCnn.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(MaxPooling2D(pool_size=(2, 2)))
    createCnn.add(Dropout(0.1))
    createCnn.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(MaxPooling2D(pool_size=(2, 2)))
    createCnn.add(Dropout(0.1))
    createCnn.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(MaxPooling2D(pool_size=(2, 2)))
    createCnn.add(Dropout(0.25))
    createCnn.add(Flatten())
    ## Neural network
    createCnn.add(Dense(128, activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Dropout(0.25))
    createCnn.add(Dense(64, activation='relu'))
    createCnn.add(BatchNormalization())
    createCnn.add(Dropout(0.4))
    createCnn.add(Dense(noClasses, activation='softmax'))
    createCnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return createCnn
    

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mfccData = np.load('spokenDigit.npz')
    batch_size = 128
    num_classes = 10
    epochs = 40
    channels = 1
    no = 3
    X_train = mfccData['X_train']
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], channels))
    X_valid = mfccData['X_valid']
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], channels))
    shapeInput = (X_train.shape[1], X_train.shape[2], channels)
    [x_train, x_test, y_train, y_test] = train_test_split(X_train, to_categorical(mfccData['Y_train']),test_size=0.1, random_state=1)
    history = AccuracyHistory()
    model = cnnModel(shapeInput, num_classes)
    #historyModel = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[history])
    #historyModel = model.fit(X_train, to_categorical(mfccData['Y_train']), batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, to_categorical(mfccData['Y_valid'])))
    
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "trainCK/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                     verbose=1, 
                                                     save_weights_only=True)
    historyModel = model.fit(X_train,
                         to_categorical(mfccData['Y_train']),
                         batch_size=batch_size, epochs=epochs,
                         verbose=1,
                         validation_data=(X_valid, to_categorical(mfccData['Y_valid'])),
                         callbacks=[cp_callback])
    
    # save the history
    f = open('history_'+str(no)+'.pckl', 'wb')
    pickle.dump(historyModel.history, f)
    f.close()
    ## Visualizing the model
    acc = historyModel.history['accuracy']
    val_acc = historyModel.history['val_accuracy']
    loss = historyModel.history['loss']
    val_loss = historyModel.history['val_loss']
    epochs_range = range(epochs)
#%%
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

#%%    
    predictions = model.predict_classes(x_test)
    Pred_report = classification_report(y_test, to_categorical(predictions))
    X_test = mfccData['X_test']
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], channels))
    predictions_2 = model.predict_classes(X_test)
    Pred_report_2 = classification_report(to_categorical(mfccData['Y_test']), to_categorical(predictions_2))
    X_valid = mfccData['X_valid']
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], channels))
    predictions_3 = model.predict_classes(X_valid)
    Pred_report_3 = classification_report(to_categorical(mfccData['Y_valid']), to_categorical(predictions_3))
    modelSave(model,2,Pred_report_3)
