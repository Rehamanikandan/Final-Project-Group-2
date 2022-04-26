import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
######################################################################
#References:
#https://www.kaggle.com/code/venkatkumar001/2-build-and-train-model-cnn-save-model
#Function definitions were refered from Exam train files
##########################################################################################

DATA_PATH = os.getcwd()+"/data_preprocess.json"
SAVED_MODEL_PATH = "model_cnn1.h5"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DECAY=1e-6
PATIENCE=20
num_classes=14
#load data
def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    print(Counter(data['labels']))
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    return X, y

#split the dataset into train,test,validation
def prepare_dataset(data_path, test_size=0.3, validation_size=0.3):
    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, y_train, X_validation, y_validation, X_test, y_test



#define the model
def model_definition(input_shape, loss="sparse_categorical_crossentropy", learning_rate=LEARNING_RATE):
        # build network architecture using convolutional layers
        model = tf.keras.Sequential()

        # 1st conv layer
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu', input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # 2nd conv layer
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # 3rd conv layer
        model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='elu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        # flatten output and feed into dense layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        tf.keras.layers.Dropout(0.5)

        # softmax output layer
        model.add(tf.keras.layers.Dense(14, activation='softmax'))

        optimiser = tf.optimizers.SGD(learning_rate=learning_rate)

        # compile model
        model.compile(optimizer=optimiser,
                      loss=loss,
                      metrics=["accuracy"])

        # print model parameters on console
        model.summary()

        return model

#train the network
def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_cnn1.h5', monitor='accuracy',
                                                     save_best_only=True)
    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback,check_point])
    return history

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    print('')
    print('')
    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()

def Build_Model():
    # generate train, validation and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)

    # create network

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    print(input_shape)
    model = model_definition(input_shape)

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set
    test_loss, test_acc= model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))


    # save model
    model.save(SAVED_MODEL_PATH)

Build_Model()