import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split

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
for i in range (len(list)):
    a = list[i].min(axis=0)
    b = list[i].max(axis=0)
    for j in range(39):
        list[i][j]=list[i][j] * (b - a) + a
X_train, X_test, y_train, y_test = train_test_split(list,labele,test_size=0.35, random_state=0)
model = RandomForestClassifier(50, max_depth=15, max_features=15)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(len([i for i in range(0,len(y_test)) if y_test[i] == preds[i]])/len(y_test))