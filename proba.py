from pydub import AudioSegment, effects
import os
lista = os.listdir('baza/')
for x in lista:
    track = AudioSegment.from_file('baza/'+x,  format= 'm4a')
    normalizedsound = effects.normalize(track) 
    normalizedsound.export('baza2/'+x, format='wav')