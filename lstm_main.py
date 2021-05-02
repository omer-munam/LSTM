import pandas as pd
import tensorflow.keras as keras
import numpy as np
from matplotlib import pyplot as plt

def build_lstm(classes):
    n_timesteps, n_features, n_outputs = 240, 4, classes.shape[0]
    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, dropout=0.5, input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def load_files():
    X = pd.read_csv('train/trainX.csv', header = None)
    y = pd.read_csv('train/trainY.csv', header = None)
    classes = pd.read_csv('classes.txt', header = None)
    return X,y,classes

def process_dataset(trainX, trainY):
    trainX = trainX.to_numpy()
    trainX = trainX.reshape(26,240,4)
    return trainX, trainY

def train_lstm(trainX, trainY, model):
    verbose, epochs, batch_size = 1, 1500, 8
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'train loss'], loc='upper right')

    # plt.subplot(2,1,2)
    # plt.plot(history.history['val_accuracy'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy/loss')
    # plt.xlabel('epoch')
    # plt.legend(['test accuracy', 'test loss'], loc='upper right')
    plt.show()
    return model

if __name__ == "__main__":
    trainX, trainY, classes = load_files()
    model = build_lstm(classes)
    trainX, trainY = process_dataset(trainX,trainY)
    # print(model.summary())
    model = train_lstm(trainX, trainY, model)
    model.save("test_model.h5")
    # print(model.summary())