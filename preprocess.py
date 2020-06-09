# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:45:35 2020

@author: Adithya Kommini
"""

import os
import shutil
from os import listdir
from os.path import isfile, join
from os import path
import librosa
import numpy as np
from keras.utils import to_categorical

def audio_to_melSpec(path,pad_len):
    y, sr = librosa.load(path, mono=True, sr=None)
    y = y[::2]
    melSpec = librosa.feature.mfcc (np.asfortranarray(y), sr=sr, n_mfcc = pad_len)
    pad_width = pad_len - melSpec.shape[1]
    melSpec = np.pad(melSpec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return melSpec

def dir_to_spectrogramMLP(audio_dir):
    labelDict = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    labels = os.listdir(audio_dir)
    specArr = []
    iolabels = []
    for label in labels:
        labelDir = audio_dir+label
        file_names = [f for f in listdir(labelDir) if isfile(join(labelDir, f)) and '.wav' in f]
        pad_len = 32
        for file_name in file_names:
            audio_path = labelDir +"\\" +file_name
            melSpec = audio_to_melSpec(audio_path,pad_len)
            #melSpec = np.asarray(melSpec)
            specArr.append(melSpec)
            iolabels.append(labelDict[label])
    return np.asarray(specArr),to_categorical(iolabels)


def splitDataset(wavDir, vList, tList):
    vListFolder = os.path.dirname(os.path.realpath(__file__)) + tList
    #os.mkdir(vListFolder)
    #vListfile = open(vList, 'r')
    with open(vList, 'r') as vListfile:
        line = vListfile.readline()
        while line:
            folderLabel = line.split('/')
            filePath = folderLabel[0]+"\\"+folderLabel[1].split('.')[0]+".wav"
            if path.exists(wavDir+filePath):
                if not os.path.isdir(vListFolder+ folderLabel[0]):
                    os.mkdir(vListFolder + folderLabel[0]+"\\")
                shutil.copyfile(wavDir+filePath,vListFolder+filePath)
                os.remove(wavDir+filePath)
    vListfile.close()
    
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    audio_dir = dir_path + "\\SpeechCommands\\"
    validList = dir_path + "\\validation_list.txt"
    testList =  dir_path + "\\testing_list.txt"
    splitDataset(audio_dir, validList, "\\validationList\\")
    splitDataset(audio_dir, testList, "\\testingList\\")
    [X_train,Y_train]=dir_to_spectrogramMLP(audio_dir)
    [X_valid,Y_valid]=dir_to_spectrogramMLP( dir_path + "\\validationList\\")
    [X_test,Y_test]=dir_to_spectrogramMLP(dir_path + "\\testingList\\")
    np.savez('spokenDigit', X_train = X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, X_test = X_test,Y_test = Y_test)
