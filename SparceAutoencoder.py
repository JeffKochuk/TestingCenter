import numpy as np
import matplotlib.pyplot as mp
import scipy.io
__author__ = 'JEFFERYK'

def normalizeData(myData):

    #Remove the Mean
    avg = np.mean(myData.ravel())
    print("Average: ",avg)
    myData -= avg

    #TRUNCATE to 3 sigmas
    std = np.std(myData)
    print("STD: ",std)
    std *= 3
    myData = np.maximum(np.minimum(myData, std),-1*std)/std

    #Rescale  from [-1,1] to [0.1,0.9]
    myData = (myData + 1) * 0.4 + 0.1

    print("New Average: ", np.mean(myData))
    print("New STD: ", np.std(myData))

    return myData



def getPatches():
    #Load Images.mat
    images = np.array(scipy.io.loadmat("IMAGES.mat")["IMAGES"])
    NUM_PICS = images.shape[2]
    PATCH_SIZE = 8
    IMAGE_SIZE = images.shape[0]
    # Unroll Images in PATCH_SIZE^2 long vectors
    images = np.array([images[j:j+PATCH_SIZE,k:k+PATCH_SIZE, i].ravel() for i in range(NUM_PICS) for j in range(0, IMAGE_SIZE, PATCH_SIZE) for k in range(0, IMAGE_SIZE, PATCH_SIZE)])
    print(images.shape)
    return images

def sigmoid(x, derivative=False):
    if derivative:
        return x * (1-x)
    else:
        return 1/(1+np.exp(-1*x))

def forward(x, L1,L2):
    return np.dot(x,W)

def cost(input,hidden,):
    cost = 0
    grad = 0




    return cost,grad



train = normalizeData(getPatches())
W = np.random.random()
for row in train:
   train(row)





