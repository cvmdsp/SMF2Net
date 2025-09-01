import os
import numpy as np
import scipy.io as sio
import MyLib as ML
import random
import cv2
from scipy.ndimage import zoom

def blur_and_downsample_hsi(hsi_data,factor):
    hsi_data = np.squeeze(hsi_data)
    height, width, bands = hsi_data.shape
    blurred_bands = []
    for band in range(bands):
        single_band = hsi_data[:, :, band]
        blurred_band = cv2.blur(single_band, (5, 5))
        blurred_bands.append(blurred_band)
    blurred_hsi = np.stack(blurred_bands, axis=2)
    downsampled_hsi = zoom(blurred_hsi, (1 / factor, 1 / factor, 1), order=1)
    return downsampled_hsi[np.newaxis,:,:,:]

def all_train_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat('.\CAVEdata/myList')
    Ind = List['list']  # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('.\CAVEdata\X/'):
        for j in range(20):
            i = Ind[0, j] - 1
            data = sio.loadmat(".\CAVEdata/X/" + files[i])
            inX = data['msi']
            allDataX.append(inX)
            data = sio.loadmat(".\CAVEdata/Y/" + files[i])
            inY = data['RGB']
            allDataY.append(inY)
    return allDataX, allDataY

def train_data_in(allX, allY, sizeI, patchNum, channel=31, dataNum=20):
    batch_X = np.zeros((patchNum, sizeI, sizeI, channel), 'f')
    batch_Y = np.zeros((patchNum, sizeI, sizeI, 3), 'f')
    batch_Z = np.zeros((patchNum, 3, 3, channel), 'f')
    for i in range(patchNum):
        ind = random.randint(0, dataNum - 1)
        X = allX[ind]
        Y = allY[ind]
        px = random.randint(0, 512 - sizeI) 
        py = random.randint(0, 512 - sizeI)
        subX = X[px:px + sizeI:1, py:py + sizeI:1, :]
        subY = Y[px:px + sizeI:1, py:py + sizeI:1, :]
    
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        for j in range(rotTimes):
            subX = np.rot90(subX)
            subY = np.rot90(subY)

        # Random vertical Flip
        for j in range(vFlip):
            subX = subX[:, ::-1, :]
            subY = subY[:, ::-1, :]

        # Random Horizontal Flip
        for j in range(hFlip):
            subX = subX[::-1, :, :]
            subY = subY[::-1, :, :]

        batch_X[i, :, :, :] = subX
        batch_Y[i, :, :, :] = subY
        batch_Z[i, :, :, :] = blur_and_downsample_hsi(subX,factor=32)

    return batch_X, batch_Y, batch_Z


def eval_data_in( batch_size):
    allX, allY = all_test_data_in()
    return train_data_in(allX, allY, 96, batch_size, 31, 12)


def all_test_data_in():
    allDataX = []
    allDataY = []
    List = sio.loadmat('.\CAVEdata\myList')
    Ind  = List['list'] # a list of rand index with frist 20 to be train data and last 12 to be test data
    for root, dirs, files in os.walk('.\CAVEdata\X/'):
           for j in range(12):
                i = Ind[0,j+20]-1
                data = sio.loadmat(".\CAVEdata\X/"+files[i])
                inX  = data['msi']
                allDataX.append(inX)
                data = sio.loadmat(".\CAVEdata/Y/"+files[i])
                inY  = data['RGB']
                allDataY.append(inY)
    return allDataX, allDataY