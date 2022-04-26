import os
#os.system("sudo pip install librosa")
import librosa
import librosa.display
import os
import json
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
############################################################################################################
#preprocessing references:
#https://www.kaggle.com/code/venkatkumar001/1-preprocessing-generate-json-file
#https://github.com/amir-jafari/Deep-Learning/blob/master/Tenflow_Advance/CNN/2_Speech_Recognition_1D/example_SpokenDigitRecognizer.py

#Documentation of Librosa:
#https://librosa.org/doc/latest/index.html
###############################################################################################################

DATASET_PATH = os.getcwd()+"/Speech_Cmd/dataset/"
JSON_PATH = "data_preprocess.json"
SAMPLES_TO_CONSIDER = 22050

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

preprocess_dataset(DATASET_PATH, JSON_PATH)
test_path=os.getcwd()+'/Speech_Cmd/test/'
test_json='data_test.json'
preprocess_dataset(test_path,test_json)

with open(os.getcwd()+'/'+JSON_PATH, "r") as f:
    data = json.load(f)
    print('Train set labels:\n',Counter(data['labels']))

with open(os.getcwd()+'/'+test_json, "r") as f:
    data1 = json.load(f)
    print('Test set labels:\n',Counter(data1['labels']))