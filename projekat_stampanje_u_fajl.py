import os
import numpy as np
import librosa
import librosa.display

FOLDER_PATH = 'ZVUKOVI/'
fileNames = os.listdir(FOLDER_PATH)
labele = []
w, h = 39, len(fileNames)
list = np.array([[0 for x in range(w)] for y in range(h)], np.float32)
for i in range(len(fileNames)):
    audio_file = fileNames[i]
    labele.append(int(audio_file[0]))
    signal, sr = librosa.load(FOLDER_PATH + audio_file)
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=39, sr=sr)
    mfccs = np.array(mfccs)
    mfccs = np.max(mfccs,axis=1)
    list[i]=mfccs
labele = np.array(labele)
sourceFile = open('mfccs.txt', 'w')
print(list, file = sourceFile)
sourceFile.close()