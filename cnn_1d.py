'''
@Author: CodeOfSH
@Github: https://github.com/CodeOfSH
@Date: 2020-05-30 15:36:40
@LastEditors: CodeOfSH
@LastEditTime: 2020-06-01 14:57:17
@Description: 
'''
import os
import scipy.io as sio
import numpy as np
import numpy.random as rng
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import Conv1D, Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers import *
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from keras.callbacks import TensorBoard

from sklearn.utils import shuffle

param_path = 'D:/Data/param_050220/'
model_path = 'D:/Codes/ml_localization/model'
motion_files = ["jwj_hori1.mat","sh_verti2.mat","shjwj_vh1.mat","jwj_hori2.mat","sh_walk1.mat","shjwj_vs1.mat","jwj_rand1.mat","sh_walk2.mat","shjwj_vv1.mat","jwj_rand2.mat","sh_walk3.mat","shjwj_vw1.mat","jwj_static1.mat","sh_walk4.mat","shjwj_vw2.mat","jwj_static2.mat","shjwj_hh1.mat","shjwj_wh1.mat","jwj_verti1.mat","shjwj_hs1.mat","shjwj_wh2.mat","jwj_verti2.mat","shjwj_hv1.mat","shjwj_ws1.mat","jwj_walk1.mat","shjwj_hw1.mat","shjwj_ws2.mat","jwj_walk2.mat","shjwj_hw2.mat","shjwj_ws3.mat","jwj_walk3.mat","shjwj_rand1.mat","shjwj_ws4.mat","jwj_walk4.mat","shjwj_rand2.mat","shjwj_wv1.mat","shjwj_rand3.mat","shjwj_wv2.mat","shjwj_sh1.mat","shjwj_ww1.mat","sh_hori1.mat","shjwj_ss1.mat","shjwj_ww2.mat","sh_hori2.mat","shjwj_ss2.mat","shjwj_ww3.mat","sh_rand1.mat","shjwj_sv1.mat","shjwj_ww4.mat","sh_rand2.mat","shjwj_sw1.mat","shjwj_ww5.mat","sh_static1.mat","shjwj_sw2.mat","shjwj_ww6.mat","sh_static2.mat","shjwj_sw3.mat","sh_verti1.mat","shjwj_sw4.mat"]
xrange = [-3.5,3.5]
yrange = [-2.5,2.5]

def read_data():
    X_train, Y_train, X_test, Y_test = [],[],[],[]
    partTrain, partPercent = True, 0.7
    allTrain, allPercent = False, 0.7
    # for fi in range(len(motion_files)):
    for fi in range(0,10):
        fname = param_path+'feature_'+motion_files[fi]
        feature = sio.loadmat(fname) #It is a dict with 2 keys: pos 936*2, pos_feature 936*600*9

        fname = param_path+'pos_'+motion_files[fi]
        gt = sio.loadmat(fname) #It is a dict with 1 key: pos_gt 6000*6

        labelMat = getLabel(xrange,yrange,gt['pos_gt']) # 0/1 matrix of size 936*600
        featureMat = feature['pos_feature']

        if partTrain:
            X_train.append(np.transpose(featureMat[:,:int(featureMat.shape[1]*partPercent),:],(1,0,2)).reshape((-1,9)))
            Y_train.append(np.transpose(labelMat[:,:int(labelMat.shape[1]*partPercent)],(1,0)).flatten())
            X_test.append(np.transpose(featureMat[:,int(featureMat.shape[1]*partPercent):,:],(1,0,2)).reshape(-1,9))
            Y_test.append(np.transpose(labelMat[:,int(labelMat.shape[1] * partPercent):],(1,0)).flatten())
        elif allTrain:
            if fi<=len(motion_files)*allPercent:
                X_train.append(np.transpose(featureMat,(1,0,2)).reshape((-1,9)))
                Y_train.append(np.transpose(labelMat,(1,0)).flatten())
            else:
                X_test.append(np.transpose(featureMat,(1,0,2)).reshape((-1,9)))
                Y_test.append(np.transpose(labelMat,(1,0)).flatten())

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    return X_train,Y_train,X_test,Y_test
    


def getLabel(xrange,yrange, pos_gt):
    nSample = pos_gt.shape[0]//10
    nY = int((yrange[1]-yrange[0])/0.2)+1
    nX = int((xrange[1]-xrange[0])/0.2)+1
    ret = np.zeros((nX*nY, nSample))
    for i in range(pos_gt.shape[0]):
        x1,y1,x2,y2 = pos_gt[i,0],pos_gt[i,1],pos_gt[i,3],pos_gt[i,4]
        if x1 !=0 or y1 !=0:
            idx = int((x1-xrange[0])/0.2)*nY+int((y1-yrange[0])/0.2)
            ret[idx,i//10]=1
        if x2!=0 or y2!=0:
            idx = int((x2 - xrange[0]) / 0.2) * nY + int((y2 - yrange[0]) / 0.2)
            ret[idx, i // 10] = 1
    return ret

def initialize_weights(shape, name=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    
def get_model(input_shape):
    # Define the tensors for the input 
    input_data = Input(input_shape)

    # 1 Dimension Conv Neural Network
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv1D(100,9, activation='relu', padding='same',
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias))
    model.add(Conv1D(60,9, activation='relu', padding='same',
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias))
    model.add((MaxPooling1D(9)))
    model.add(Conv1D(30,3, activation='relu', padding='same',
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias))
    model.add(Conv1D(30,3, activation='relu', padding='same',
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid',
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias))
    return model

def get_batch(X_train_0, X_train_1, batch_size):
    len_train_0=X_train_0.shape[0]
    len_train_1=X_train_1.shape[0]

    new_input = np.zeros((batch_size,9,1))
    new_label = np.zeros((batch_size,))

    new_label[batch_size//2:]=1

    for i in range(batch_size):
        if i < batch_size//2:
            index = rng.randint(0,len_train_0)
            new_input[i,:]=X_train_0[index].reshape(9,1)
        else:
            index = rng.randint(0,len_train_1)
            new_input[i,:]=X_train_1[index].reshape(9,1)

    # shuffle the generated data
    new_input, new_label = shuffle(new_input, new_label)

    return new_input, new_label

def test_task(model, X_test, Y_test, mode=0):
    print("---Evaluating model on test tasks---")
    if mode == 0:
        probs = model.predict(X_test)
        probs=np.squeeze(probs)
        print(probs)
        n_correct=np.sum((probs>0.5)==Y_test)
        percent_correct=n_correct/(Y_test.shape[0])*100.0
        print("Got an average of {}% learning accuracy with correnct num of {}".format(percent_correct, n_correct))
        return percent_correct
    else:
        FF,FT,TF,TT = 0,0,0,0
        Y_pred = model.predict(X_test)
        print(Y_pred)
        for i in range(Y_pred.shape[0]):
            if Y_test[i]==0:
                if Y_pred[i] < 0.5:
                    FF+=1
                else:
                    FT+=1
            else:
                if Y_pred[i] < 0.5:
                    TF += 1
                else:
                    TT += 1
        print('FF:{},FT:{},TF:{},TT:{}'.format(FF,FT,TF,TT))
    

if __name__=='__main__':
    
    X_train,Y_train,X_test,Y_test = read_data()
    X_test=np.reshape(X_test,(-1,9,1))

    X_train_0=X_train[Y_train==0]
    X_train_1=X_train[Y_train==1]

    model = get_model((9,1))
    model.summary()

    optimizer= Adam(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    batch_size = 1024
    iteration = 50000
    record_every = 100
    best = -1

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()

    # history = model.fit(X_train, Y_train, epochs=100, batch_size=512, validation_split=0.2)
    # val_acc = test_task(model, X_test, Y_test)
    # test_task(model, X_test, Y_test, mode=1)

    losses=[]
    for i in range(iteration):
        (inputs, labels) = get_batch(X_train_0, X_train_1, batch_size)
        loss = model.train_on_batch(inputs, labels)
        if i % record_every ==0:
            print("Step: {0} Loss: {1}".format(i,loss))
            losses.append(loss)
    
    val_acc = test_task(model, X_test, Y_test, mode=0)
    test_task(model, X_test, Y_test, mode=1)

    plt.plot(losses, 'bo', label='Training loss')
    plt.show()