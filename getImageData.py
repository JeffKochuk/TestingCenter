import numpy as np
import random as rand
import scipy.io
__author__ = 'JEFFERYK'

def normalizeData(myData):

    #Remove the Mean
    avg = np.mean(myData,axis=0)
    myData -= avg

    #TRUNCATE to 3 sigmas
    std = np.std(myData)
    std *= 3
    myData = np.maximum(np.minimum(myData, std),-1*std)/std

    #Rescale  from [-1,1] to [0.1,0.9]
    myData = (myData + 1) * 0.4 + 0.1

    return myData

def getPatches(n):
    images = np.array(scipy.io.loadmat("IMAGES.mat")["IMAGES"])
    NUM_PICS = images.shape[2]
    PATCH_SIZE = 8
    IMAGE_SIZE = images.shape[0]
    patches = np.zeros((n,PATCH_SIZE*PATCH_SIZE))
    for i in range(n):
        x,y,z = getrands(NUM_PICS,PATCH_SIZE,IMAGE_SIZE)
        patches[i,:] = images[y:y+PATCH_SIZE,z:z+PATCH_SIZE, x].ravel()
    return normalizeData(patches)


def getrands(NUM_PICS, PATCH_SIZE, IMAGE_SIZE):
    return rand.randint(0,NUM_PICS-1),rand.randint(0,IMAGE_SIZE-PATCH_SIZE-1),rand.randint(0,IMAGE_SIZE-PATCH_SIZE-1)
