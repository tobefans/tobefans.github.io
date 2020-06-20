#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:06:21 2020

@author: tobefans
"""

from matplotlib import pyplot as plt
import librosa
import librosa.display


# Load a wav file
y, sr = librosa.load('./test.wav', sr=None)
# plot a wavform
plt.figure()
librosa.display.waveplot(y, sr)
# plt.plot(y)
plt.title('wavform')
plt.show()

# extract mel spectrogram feature
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, win_length=1024, hop_length=512, n_mels=128, power=2.0)
# convert to log scale
logmelspec = librosa.power_to_db(melspec)
# plot mel spectrogram
plt.figure()
librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.title('spectrogram')
plt.show()


# aspect设为auto即可自动拉宽图
plt.imshow(logmelspec, origin="lower", cmap = "jet", aspect = "auto", interpolation = "none")
plt.show()
plt.xticks([])
plt.yticks([])
plt.savefig('specgram.png',bbox_inches='tight',pad_inches=0.0)
plt.close()


# import numpy as np
# from scipy import io
# import matplotlib.pyplot as plt

# Fs, x =  io.wavfile.read('test.wav')
# wave = np.array(x[:,0], dtype = "float")

# frame_len = 1000
# frame_off = frame_len // 2    # 非重叠点数
# specg_len = 1024

# # 可以想象1是代表第一帧，然后第二帧结尾超出第一帧frame_off个点，第三帧再超出第二帧frame_off个点，总共第二帧到最后一帧共有(wave.size - frame_len) // frame_off 帧
# frame_num = (wave.size - frame_len) // frame_off + 1
# # 生成汉明窗
# hamwindow = np.hamming(frame_len)
# specg = np.zeros((frame_num, specg_len // 2 + 1))
# z = np.zeros(specg_len - frame_len)

# for idx in range(frame_num):
#     base = idx * frame_off
#     frame = wave[base: base + frame_len]            # 分帧
#     frame = np.append(frame * hamwindow, z)         # 加窗
#     specg[idx:] = np.log10(np.abs(np.fft.rfft(frame))) # FFT，返回幅度谱

# specg = np.transpose(specg)
# io.savemat('specgram.mat', {'specg':specg})

# # aspect设为auto即可自动拉宽图
# plt.imshow(specg, origin="lower", cmap = "jet", aspect = "auto", interpolation = "none")