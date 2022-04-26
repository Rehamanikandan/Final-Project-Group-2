import json
import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

#https://www.tensorflow.org/tutorials/audio/simple_audio

DATA_PATH = os.getcwd()+"/data_test.json"
SAVED_MODEL_PATH = "model_cnn.h5"
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
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
X,y=load_data(DATA_PATH)
#make predictions using the model
model=tf.keras.models.load_model(SAVED_MODEL_PATH)
predictions=model.predict(X)
y_pred = np.argmax(predictions, axis=1)


test_acc = sum(y_pred == y) / len(y)
print(f'Test set accuracy: {test_acc:.0%}')
col_name=['bird','cat','dog','down','go','happy','left','no','off','right','stop','tree','up','yes']
confusion_mtx = tf.math.confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=col_name,
            yticklabels=col_name,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

