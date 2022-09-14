import os
import scipy.io 
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
model2 = XGBClassifier(objective='multiclass:softmax', learning_rate = 0.311,max_depth = 3, n_estimators = 500)
model2.fit(X_train, y_train)
preds = model2.predict(X_test)
x = [0,1,2,3,4,5,6,7,8,9]
import seaborn as sns
labels = x 
column = [f'Predvidjeno {label}'for label in labels]
ind = [f'Pravo {label}'for label in labels]
table = pd.DataFrame(confusion_matrix(y_test,preds), 
                    columns = column, index = ind)
ax = sns.heatmap(table, annot=True, fmt='d',cmap="YlGnBu",linewidths=.5)
plt.show()