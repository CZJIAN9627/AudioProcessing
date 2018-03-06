#-*- coding:utf-8 -*-
#!/usr/bin/python

import wave
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def wavread(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData, dtype=np.int16)
    f.close()
    waveData = waveData * 1.0 / (max(abs(waveData)))
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    return waveData, framerate

def audio2frame(signal, frame_length=512, frame_step=128, winfunc=lambda x:np.ones((x,))):
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    if signal_length <= frame_length:
        frames_num = 1
    else:
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num + 1) * frame_step + frame_length)
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1)) + np.tile(
        np.arange(0, frames_num * frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    win = np.tile(winfunc(frame_length), (frames_num, 1))
    return frames * win

def cepstral_smoothing(frames, NFFT=512):
    spectrum = np.absolute(np.fft.rfft(frames, NFFT))
    pow_spectrum = (1/NFFT) * np.square(spectrum)
    pow_spectrum[pow_spectrum < 1e-30] = 1e-30
    log_pow_spectrum = 10 * np.log10(pow_spectrum)
    cepstrum = np.fft.fft(log_pow_spectrum)
    height, width = cepstrum.shape
    window_width = 8 #
    one = np.ones((height, window_width))
    zero = np.zeros((height, width - window_width))
    window = np.concatenate((one, zero), axis=1)
    cepstrum_modified = cepstrum * window
    spec_envelope = np.fft.ifft(cepstrum_modified)
    result = np.real(spec_envelope)
    return result

import operator
from scipy.signal import argrelmax

def get_formants(result):
    nframes = result.shape[0]
    all_peaks = []
    for i in np.arange(nframes):
        peak_dict = {}
        peak_position = argrelmax(result[i])
        # to ensure 4 formants of each frame
        if len(peak_position[0]) < 4:
            continue
        peak_value = result[i][peak_position]
        for j in np.arange(len(peak_position[0])):
            peak_dict[result[i][peak_position][j]] = peak_position[0][j]
        sorted_peak_dict = dict(sorted(peak_dict.items(),
                                       key=operator.itemgetter(1)))
        numOfpeak = 0
        peaks = []
        for key in sorted_peak_dict:
            numOfpeak += 1
            if numOfpeak > 4:
                break
            peaks.append(sorted_peak_dict[key])
        all_peaks.append(peaks)
    all_formants = np.array(all_peaks)
    all_formants.sort(axis=0)
    size = all_formants.shape[0]
    if size % 2 == 0:
        formants = all_formants[int((size/2)-1)]
    else:
        formants = all_formants[int((size-1)/2)]
    return formants

if __name__ == '__main__':
    filepath = 'E:/MACHINE LEARNING/signals/signals_1/'
    dirname = os.listdir(filepath)
    filename = filepath + dirname[10]
    data, framerate = wavread(filename)
    frames = audio2frame(data[0], 512, 128)
    result = cepstral_smoothing(frames, 512)
    # print(result.shape)
    # plt.plot(result[100]);plt.show()
    formants = get_formants(result)
    print('Congratulations!')

