import os
#os.system("sudo pip install librosa")
import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

############################################################################################################
#Analysis references:
#https://www.kaggle.com/code/venkatkumar001/1-preprocessing-generate-json-file

#Documentation of Librosa:
#https://librosa.org/doc/latest/index.html
###############################################################################################################

#Load one data from subclasses of data and visualize
bird,sr=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/bird/00b01445_nohash_0.wav")
cat,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/cat/004ae714_nohash_0.wav")
dog,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/dog/00b01445_nohash_0.wav")
down,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/down/00176480_nohash_0.wav")
go,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/go/004ae714_nohash_0.wav")
happy,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/happy/012c8314_nohash_0.wav")
left,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/left/00176480_nohash_0.wav")
no,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/no/012c8314_nohash_0.wav")
off,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/off/00176480_nohash_0.wav")
right,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/right/00b01445_nohash_0.wav")
stop,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/stop/004ae714_nohash_0.wav")
tree,_= librosa.load(os.getcwd()+"/Speech_Cmd/dataset/tree/00b01445_nohash_0.wav")
up,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/up/0137b3f4_nohash_3.wav")
yes,_=librosa.load(os.getcwd()+"/Speech_Cmd/dataset/yes/004ae714_nohash_0.wav")

#audio signal duration
sample_duration = 1/sr
duration = sample_duration * len(bird)
print(f"Duration of sample is:{duration: .6f} seconds")

#visualize the input
col=[bird,cat,dog,down,go,happy,left,no,off,right,stop,tree,up,yes]
col_name=['bird','cat','dog','down','go','happy','left','no','off','right','stop','tree','up','yes']
z=0
plt.figure(figsize=(15, 17))
for i in range(1,15):
        plt.subplot(7,2,i)
        librosa.display.waveshow(col[z],alpha=0.5)
        plt.title(f'Waveform of {col_name[z]}')
        plt.ylim((-1, 1))
        z+=1
plt.tight_layout()
plt.show()

#FFT
plt.figure(figsize=(15, 17))
z=0
for i in range(1,15):
        plt.subplot(7,2,i)
        ffts= np.fft.fft(col[z])
        magnitude_d = np.abs(ffts)
        frequency_d = np.linspace(0, sr, len(magnitude_d))
        plt.plot(frequency_d, magnitude_d)
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title(f'FFT of {col_name[z]}')
        z+=1
plt.tight_layout()
plt.show()

#STFT
n_fft = 2048 #no.of.sample
hop_length = 512 #amount of shift h-fouriertransform
plt.figure(figsize=(15, 17))
z=0
for i in range(1,15):
        plt.subplot(7,2,i)
        stft_d = librosa.core.stft(col[z], hop_length=hop_length, n_fft=n_fft)
        spectrogram_d = np.abs(stft_d)
        # convert viewable form of low point
        log_spectrogram_d = librosa.amplitude_to_db(spectrogram_d)
        librosa.display.specshow(log_spectrogram_d, sr=sr, hop_length=hop_length)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.title(f'STFT of {col_name[z]}')
        z+=1
plt.tight_layout()
plt.show()

#MFCCS
plt.figure(figsize=(15, 17))
z=0
for i in range(1,15):
        plt.subplot(7,2,i)
        MFFCS_d = librosa.feature.mfcc(col[z],n_fft=n_fft,hop_length=hop_length,n_mfcc=13)
        librosa.display.specshow(MFFCS_d, sr=sr, hop_length=hop_length)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.title(f'MFCC of {col_name[z]}')
        z+=1
plt.tight_layout()
plt.show()