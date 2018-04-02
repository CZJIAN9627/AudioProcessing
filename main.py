#-*- coding:utf-8 -*-
#!/usr/bin/python

import os
import gc #
import numpy as np
from formants import *

filepath = 'E:/MACHINE LEARNING/signals/signals_1/'
dirname = os.listdir(filepath)
numOffile = len(dirname)

formant_file = open(r'D:/Documents/Python/Audio Signals Processing/record.txt','a+')
for i in np.arange(numOffile):
    filename = filepath + dirname[i]
    data, framerate = wavread(filename)
    frames = audio2frame(data[0], 512, 128)
    result = cepstral_smoothing(frames, 512)
    formants = get_formants(result)
    # to write list into file
    for j in np.arange(len(formants)):
        formant_file.write(str(formants[j])+' ')
    formant_file.write('\n')
    gc.collect() #
    print('Done!Continue...')
formant_file.close()

print('Congratulations!')
